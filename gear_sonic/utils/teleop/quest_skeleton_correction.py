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
    (+0.742367672, -0.070341054, -0.017498777, -0.666060184),  # 20 LEFT_WRIST      axis 14d
    (+0.542692508, -0.009660575, -0.128577186, +0.829975555),  # 21 RIGHT_WRIST     axis 14d
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
    (-0.991547347, -0.049393051, +0.066475985, -0.099875564),  # 20 LEFT_WRIST      axis 14d
    (-0.283646282, +0.012514153, +0.023354725, +0.958562851),  # 21 RIGHT_WRIST     axis 14d
    (+0.000000001, +0.000000001, -0.000000000, +1.000000000),  # 22 LEFT_HAND       no motion
    (+0.000000000, +0.000000000, +0.000000000, +1.000000000),  # 23 RIGHT_HAND      no motion  [one-sided]
)

# Joints fitted from static poses only (<15 deg of rotation across all captures),
# so less well determined than the rest. No axis-mapping risk: they do not move.
STATIC_ONLY_JOINTS = (6, 7, 8, 9, 10, 11, 13, 14, 22)

# Quest infers legs from vision; degrades on deep flexion (see docstring).
INFERRED_LEG_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)
