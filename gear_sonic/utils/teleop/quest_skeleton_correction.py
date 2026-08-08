"""Per-headset wrist bias for the G1, in commanded joint space.

The Quest-to-Pico skeleton orientation correction that used to live here now
lives upstream in isaacteleop, as
``isaacteleop.retargeting_engine.utilities.correct_body_orientations`` (see
NVIDIA/IsaacTeleop#921), and is applied in the reader. Its derivation, tables
and residuals are recorded there.

What stays here is the wrist bias, which is not part of that correction and does
not belong upstream: it is expressed in G1 wrist joint commands (roll, pitch,
yaw), so it is specific to this robot rather than to the headset's skeleton
convention. Re-expressing it as a skeleton-space rotation is what it would take
to move it, and that has not been done.
"""

# ---------------------------------------------------------------------------
# Wrist bias, radians, added to the commanded G1 wrist joints as
# (roll, pitch, yaw) per side.
#
# Measured at `01_wrist_neutral` -- forearms forward, elbows 90, wrists straight
# and relaxed -- where all three channels should read the same on both headsets.
# They did not:
#
#     channel     Pico     Quest    bias applied
#     L roll     -33.6     -24.5       +24.24   (zeroed, not matched)
#     L pitch     +7.2      +7.1       +11.44
#     L yaw      +15.2     +44.1       -29.88
#     R roll     +22.3     +19.7       -19.47   (zeroed, not matched)
#     R pitch     +9.4      +9.4       +16.87
#     R yaw      -19.9     -48.2       +28.13
#
# ROLL is the exception: it is driven to ZERO rather than matched to the Pico.
# The Pico commands -33.6 / +22.2 deg of wrist roll at a neutral wrist, which
# renders as visibly rolled palms, and it is asymmetric -- the left sits 11.4 deg
# further round than the right. An operator sees exactly that: both palms rolled
# up, the left more. Matching the Pico faithfully reproduced the flaw, so roll
# alone departs from the reference. Because a constant bias shifts the operating
# point without compressing travel, this costs nothing in responsiveness: roll
# span across the wrist sweep is 91 / 86 deg either way.
#
# Caveat: zero commanded roll is a HYPOTHESIS about where the robot's neutral
# lies, not a measurement. There is no ground truth here for "thumbs facing each
# other" in joint terms. If a trial still shows residual roll, adjust these two
# numbers by the observed amount -- the rest of the table does not depend on them.
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
    "quest": ((+0.423036, +0.199678, -0.521456),
              (-0.339744, +0.294511, +0.490959)),
}
