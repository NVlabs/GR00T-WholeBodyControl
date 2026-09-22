"""Unitree G1 upper-body motor indices and display groups."""

UPPER_BODY_JOINTS = (
    (12, "waist_yaw"),
    (13, "waist_roll"),
    (14, "waist_pitch"),
    (15, "left_shoulder_pitch"),
    (16, "left_shoulder_roll"),
    (17, "left_shoulder_yaw"),
    (18, "left_elbow"),
    (19, "left_wrist_roll"),
    (20, "left_wrist_pitch"),
    (21, "left_wrist_yaw"),
    (22, "right_shoulder_pitch"),
    (23, "right_shoulder_roll"),
    (24, "right_shoulder_yaw"),
    (25, "right_elbow"),
    (26, "right_wrist_roll"),
    (27, "right_wrist_pitch"),
    (28, "right_wrist_yaw"),
)

UPPER_BODY_INDICES = tuple(index for index, _ in UPPER_BODY_JOINTS)
UPPER_BODY_NAMES = tuple(name for _, name in UPPER_BODY_JOINTS)

# Indices below address the compact 17-element upper-body arrays, not LowState.
PLOT_GROUPS = (
    ("TORSO", tuple(range(0, 3))),
    ("LEFT ARM", tuple(range(3, 10))),
    ("RIGHT ARM", tuple(range(10, 17))),
)
