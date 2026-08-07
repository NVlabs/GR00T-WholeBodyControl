from .base import SkeletonBase

# If a joint's channel is not in this list, it is a dead joint that is not activated
# This is the list of channels which does not include hand motions
ACTIVATED_JOINTS_CHANNELS_IN_G1 = [
    # pelvis
    "pelvis_skel.translateX",
    "pelvis_skel.translateY",
    "pelvis_skel.translateZ",
    "pelvis_skel.rotateX",
    "pelvis_skel.rotateY",
    "pelvis_skel.rotateZ",
    # right hip & right leg & right foot
    "right_hip_pitch_skel.rotateX",
    "right_hip_roll_skel.rotateZ",
    "right_hip_yaw_skel.rotateY",
    "right_knee_skel.rotateX",
    "right_ankle_pitch_skel.rotateX",
    "right_ankle_roll_skel.rotateZ",
    # waist
    "waist_yaw_skel.rotateY",
    "waist_roll_skel.rotateZ",
    "waist_pitch_skel.rotateX",
    # right shoulder & right arm & right hand
    "right_shoulder_pitch_skel.rotateX",
    "right_shoulder_roll_skel.rotateZ",
    "right_shoulder_yaw_skel.rotateY",
    "right_elbow_skel.rotateX",
    # left shoulder & left arm & left hand
    "left_shoulder_pitch_skel.rotateX",
    "left_shoulder_roll_skel.rotateZ",
    "left_shoulder_yaw_skel.rotateY",
    "left_elbow_skel.rotateX",
    # left hip & left leg & left foot
    "left_hip_pitch_skel.rotateX",
    "left_hip_roll_skel.rotateZ",
    "left_hip_yaw_skel.rotateY",
    "left_knee_skel.rotateX",
    "left_ankle_pitch_skel.rotateX",
    "left_ankle_roll_skel.rotateZ",
]

# in total we have 32 joints; other than pelvis, each joint is a hinge joint with 1 degree of freedom (including the
# dead joints)
COMPLETE_JOINT_LIST_IN_G1 = [
    "pelvis_skel",
    # left hip & left leg & left foot
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    # right hip & right leg & right foot
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    # waist
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    # left shoulder & left arm & left hand
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    # right shoulder & right arm & right hand
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
]


class G1Skeleton(SkeletonBase):
    bone_order_names_with_parents = []
    name = "g1skel"
    right_hand_joint_names = ["right_hand_roll_skel"]
    left_hand_joint_names = ["left_hand_roll_skel"]
    hip_joint_names = [
        "right_hip_pitch_skel",
        "left_hip_pitch_skel",
    ]  # used to calculate root orientation, only need 1 pair of hip joints

    def get_skel_slice(self, skeleton: SkeletonBase):
        """Return a slice element so that we can slice the input data into our current skeleton."""
        try:
            skel_slice = [skeleton.bone_index[x] for x in self.bone_order_names]
        except KeyError:
            raise ValueError(
                "The current skeleton contain joints that are not in the input"
            )
        return skel_slice


class G1Skeleton32(G1Skeleton):
    """This is the full skeleton with all 32 joints.

    but no toe joints.
    """

    name = "g1skel32"
    right_foot_joint_names = ["right_ankle_pitch_skel", "right_ankle_roll_skel"]
    left_foot_joint_names = ["left_ankle_pitch_skel", "left_ankle_roll_skel"]

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        # left hip & left leg & left foot
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        # right hip & right leg & right foot
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        # waist
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_roll_skel", "waist_yaw_skel"),
        ("waist_pitch_skel", "waist_roll_skel"),
        # left shoulder & left arm & left hand
        ("left_shoulder_pitch_skel", "waist_pitch_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_roll_skel", "left_elbow_skel"),
        ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
        ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
        ("left_hand_roll_skel", "left_wrist_yaw_skel"),
        # right shoulder & right arm & right hand
        ("right_shoulder_pitch_skel", "waist_pitch_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_roll_skel", "right_elbow_skel"),
        ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
        ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
        ("right_hand_roll_skel", "right_wrist_yaw_skel"),
    ]


class G1Skeleton34(G1Skeleton):
    """This is the full skeleton with all 32 joints, + 2 dummy toe joints."""

    name = "g1skel34"

    bone_order_names_with_parents = [
        ("pelvis_skel", None),
        # left hip & left leg & left foot
        ("left_hip_pitch_skel", "pelvis_skel"),
        ("left_hip_roll_skel", "left_hip_pitch_skel"),
        ("left_hip_yaw_skel", "left_hip_roll_skel"),
        ("left_knee_skel", "left_hip_yaw_skel"),
        ("left_ankle_pitch_skel", "left_knee_skel"),
        ("left_ankle_roll_skel", "left_ankle_pitch_skel"),
        ("left_toe_base", "left_ankle_roll_skel"),
        # right hip & right leg & right foot
        ("right_hip_pitch_skel", "pelvis_skel"),
        ("right_hip_roll_skel", "right_hip_pitch_skel"),
        ("right_hip_yaw_skel", "right_hip_roll_skel"),
        ("right_knee_skel", "right_hip_yaw_skel"),
        ("right_ankle_pitch_skel", "right_knee_skel"),
        ("right_ankle_roll_skel", "right_ankle_pitch_skel"),
        ("right_toe_base", "right_ankle_roll_skel"),
        # waist
        ("waist_yaw_skel", "pelvis_skel"),
        ("waist_roll_skel", "waist_yaw_skel"),
        ("waist_pitch_skel", "waist_roll_skel"),
        # left shoulder & left arm & left hand
        ("left_shoulder_pitch_skel", "waist_pitch_skel"),
        ("left_shoulder_roll_skel", "left_shoulder_pitch_skel"),
        ("left_shoulder_yaw_skel", "left_shoulder_roll_skel"),
        ("left_elbow_skel", "left_shoulder_yaw_skel"),
        ("left_wrist_roll_skel", "left_elbow_skel"),
        ("left_wrist_pitch_skel", "left_wrist_roll_skel"),
        ("left_wrist_yaw_skel", "left_wrist_pitch_skel"),
        ("left_hand_roll_skel", "left_wrist_yaw_skel"),
        # right shoulder & right arm & right hand
        ("right_shoulder_pitch_skel", "waist_pitch_skel"),
        ("right_shoulder_roll_skel", "right_shoulder_pitch_skel"),
        ("right_shoulder_yaw_skel", "right_shoulder_roll_skel"),
        ("right_elbow_skel", "right_shoulder_yaw_skel"),
        ("right_wrist_roll_skel", "right_elbow_skel"),
        ("right_wrist_pitch_skel", "right_wrist_roll_skel"),
        ("right_wrist_yaw_skel", "right_wrist_pitch_skel"),
        ("right_hand_roll_skel", "right_wrist_yaw_skel"),
    ]
    right_foot_joint_names = ["right_ankle_roll_skel", "right_toe_base"]
    left_foot_joint_names = ["left_ankle_roll_skel", "left_toe_base"]
