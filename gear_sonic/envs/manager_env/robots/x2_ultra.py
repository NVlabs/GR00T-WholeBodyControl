from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# Agibot X2 Ultra motor parameters.
# Exact rotor inertia (armature) values are not available from the vendor;
# using H2 motor constants as starting estimates grouped by torque class.
# These MUST be tuned once real motor datasheets are obtained.
ARMATURE_HIP_KNEE = 0.025101925  # 120 N-m class (hip pitch/roll/yaw, knee)
ARMATURE_WAIST_YAW = 0.010177520  # 120 N-m waist yaw
ARMATURE_WAIST_PR = 0.003609725  # 48 N-m waist pitch/roll
ARMATURE_ANKLE = 0.003609725  # 36/24 N-m ankle
ARMATURE_SHOULDER_ELBOW = 0.003609725  # 36/24 N-m shoulder/elbow
ARMATURE_WRIST = 0.00425  # 4.8 N-m wrist pitch/roll
ARMATURE_WRIST_YAW = 0.003609725  # 24 N-m wrist yaw
ARMATURE_HEAD = 0.00425  # 2.6/0.6 N-m head

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_WAIST_YAW = ARMATURE_WAIST_YAW * NATURAL_FREQ**2
STIFFNESS_WAIST_PR = ARMATURE_WAIST_PR * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2
STIFFNESS_SHOULDER_ELBOW = ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ**2
STIFFNESS_WRIST = ARMATURE_WRIST * NATURAL_FREQ**2
STIFFNESS_WRIST_YAW = ARMATURE_WRIST_YAW * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_WAIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_YAW * NATURAL_FREQ
DAMPING_WAIST_PR = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_PR * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_SHOULDER_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ
DAMPING_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_WRIST * NATURAL_FREQ
DAMPING_WRIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WRIST_YAW * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ

# Body names in IsaacLab BFS traversal order (32 bodies including root pelvis).
# Derived by BFS traversal of the URDF kinematic tree, skipping fixed-joint-only links.
X2_ULTRA_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "head_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "head_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]

# DOF index mappings between IsaacLab and MuJoCo orderings (31 DOF).
# isaaclab_to_mujoco[i] = MuJoCo index of the joint at IsaacLab index i.
X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 29, 4, 10,
    16, 23, 30, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11,
    16, 21, 23, 25, 27, 29, 12, 17, 22, 24, 26, 28, 30, 13, 18,
]

# Body index mappings between IsaacLab and MuJoCo orderings (32 bodies).
X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 30, 5,
    11, 17, 24, 31, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 21, 3, 6, 9,
    12, 17, 22, 24, 26, 28, 30, 13, 18, 23, 25, 27, 29, 31, 14, 19,
]

X2_ULTRA_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": X2_ULTRA_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY,
}


X2_ULTRA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/x2_ultra/x2_ultra.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # X2 Ultra pelvis height is ~0.68m in MJCF default pose;
        # spawn slightly higher to avoid ground clipping with bent knees.
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": -0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 11.936,
                ".*_hip_roll_joint": 11.936,
                ".*_hip_pitch_joint": 11.936,
                ".*_knee_joint": 11.936,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_HIP_KNEE,
                ".*_hip_roll_joint": STIFFNESS_HIP_KNEE,
                ".*_hip_yaw_joint": STIFFNESS_HIP_KNEE,
                ".*_knee_joint": STIFFNESS_HIP_KNEE,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_KNEE,
                ".*_hip_roll_joint": DAMPING_HIP_KNEE,
                ".*_hip_yaw_joint": DAMPING_HIP_KNEE,
                ".*_knee_joint": DAMPING_HIP_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP_KNEE,
                ".*_hip_roll_joint": ARMATURE_HIP_KNEE,
                ".*_hip_yaw_joint": ARMATURE_HIP_KNEE,
                ".*_knee_joint": ARMATURE_HIP_KNEE,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.088,
                ".*_ankle_roll_joint": 15.077,
            },
            stiffness=STIFFNESS_ANKLE,
            damping=DAMPING_ANKLE,
            armature=ARMATURE_ANKLE,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=11.936,
            stiffness=STIFFNESS_WAIST_YAW,
            damping=DAMPING_WAIST_YAW,
            armature=ARMATURE_WAIST_YAW,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.088,
            stiffness=STIFFNESS_WAIST_PR,
            damping=DAMPING_WAIST_PR,
            armature=ARMATURE_WAIST_PR,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={
                "head_yaw_joint": 2.6,
                "head_pitch_joint": 0.6,
            },
            velocity_limit_sim={
                "head_yaw_joint": 6.019,
                "head_pitch_joint": 6.28,
            },
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HEAD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_roll_joint": 4.8,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.088,
                ".*_shoulder_roll_joint": 13.088,
                ".*_shoulder_yaw_joint": 15.077,
                ".*_elbow_joint": 15.077,
                ".*_wrist_yaw_joint": 15.077,
                ".*_wrist_pitch_joint": 4.188,
                ".*_wrist_roll_joint": 4.188,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_elbow_joint": STIFFNESS_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
                ".*_wrist_pitch_joint": STIFFNESS_WRIST,
                ".*_wrist_roll_joint": STIFFNESS_WRIST,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": DAMPING_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": DAMPING_SHOULDER_ELBOW,
                ".*_elbow_joint": DAMPING_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
                ".*_wrist_pitch_joint": DAMPING_WRIST,
                ".*_wrist_roll_joint": DAMPING_WRIST,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_shoulder_roll_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_shoulder_yaw_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_elbow_joint": ARMATURE_SHOULDER_ELBOW,
                ".*_wrist_yaw_joint": ARMATURE_WRIST_YAW,
                ".*_wrist_pitch_joint": ARMATURE_WRIST,
                ".*_wrist_roll_joint": ARMATURE_WRIST,
            },
        ),
    },
)

# Action scale: effort_limit / stiffness * 0.25 (same formula as H2)
X2_ULTRA_ACTION_SCALE = {}
for a in X2_ULTRA_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            X2_ULTRA_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
