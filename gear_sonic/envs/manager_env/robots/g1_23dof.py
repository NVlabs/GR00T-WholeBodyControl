"""Native Unitree G1 23-DoF rev-1.0 IsaacLab configuration."""

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from gear_sonic.envs.manager_env.robots.g1 import (
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)
from gear_sonic.utils.g1_23dof_contract import (
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
)

# Isaac Lab 2.3's URDF converter expects an absolute asset path.  Resolving
# from this module also keeps first spawn independent of the launch directory.
ASSET_DIR = Path(__file__).resolve().parents[3] / "data" / "robots" / "g1"

# Root plus 23 actuated child bodies, in IsaacLab articulation order.
G1_23DOF_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]

G1_23DOF_ISAACLAB_TO_MUJOCO_DOF = list(ISAACLAB_TO_MUJOCO_DOF)
G1_23DOF_MUJOCO_TO_ISAACLAB_DOF = list(MUJOCO_TO_ISAACLAB_DOF)
G1_23DOF_ISAACLAB_TO_MUJOCO_BODY = [0] + [
    index + 1 for index in G1_23DOF_ISAACLAB_TO_MUJOCO_DOF
]
G1_23DOF_MUJOCO_TO_ISAACLAB_BODY = [0] + [
    index + 1 for index in G1_23DOF_MUJOCO_TO_ISAACLAB_DOF
]

G1_23DOF_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": G1_23DOF_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": G1_23DOF_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": G1_23DOF_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": G1_23DOF_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": G1_23DOF_MUJOCO_TO_ISAACLAB_BODY,
}

G1_23DOF_REV_1_0_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(ASSET_DIR / "g1_23dof_rev_1_0.urdf"),
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
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
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
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_yaw_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_yaw_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=STIFFNESS_5020,
            damping=DAMPING_5020,
            armature=ARMATURE_5020,
        ),
    },
)

# Unitree rev-1.0 deployment scales. In particular, ankles use 0.44 even
# though their simulation torque limit is 35 Nm; deriving scale from that
# clamp would incorrectly produce about 0.307.
G1_23DOF_ACTION_SCALE = {
    ".*_hip_pitch_joint": 0.55,
    ".*_hip_roll_joint": 0.35,
    ".*_hip_yaw_joint": 0.55,
    ".*_knee_joint": 0.35,
    ".*_ankle_pitch_joint": 0.44,
    ".*_ankle_roll_joint": 0.44,
    "waist_yaw_joint": 0.55,
    ".*_shoulder_pitch_joint": 0.44,
    ".*_shoulder_roll_joint": 0.44,
    ".*_shoulder_yaw_joint": 0.44,
    ".*_elbow_joint": 0.44,
    ".*_wrist_roll_joint": 0.44,
}
