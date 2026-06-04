/**
 * @file policy_parameters.hpp
 * @brief Motor constants, PID gains, joint mappings, action scales, and default
 *        standing angles for the H2 31-DOF policy.
 *
 * ## Joint Ordering
 *
 * Two ordering conventions are used at the H2 deploy boundary:
 *  - **Motor-slot order** – DDS LowCmd/LowState order and the sim2sim WBC
 *    boundary order. This is the order of kps/kds/default_angles/action_scale.
 *  - **Policy/IsaacLab order** – ONNX action and joint-observation order.
 *
 * Prefer the explicit aliases `H2_POLICY_TO_MOTOR` and `H2_MOTOR_TO_POLICY`.
 * The older names `mujoco_to_isaaclab` and `isaaclab_to_mujoco` are kept only
 * for existing call sites and have the same operational values as the aliases.
 *
 * ## PID Gain Computation
 *
 * Stiffness (Kp) and damping (Kd) values are computed from motor armature
 * constants using a second-order critically-damped model:
 *   - stiffness = armature × ω²   (ω = 10 Hz × 2π)
 *   - damping   = 2 × ζ × armature × ω   (ζ = 2.0)
 *
 * ## Action Scaling
 *
 * Policy actions are scaled by:
 *   action_scale = 0.25 × effort_limit / stiffness
 *
 * The final joint target is: target = action × action_scale + default_angle.
 */

#ifndef POLICY_PARAMETERS_HPP
#define POLICY_PARAMETERS_HPP

#include <array>
#include <vector>

static constexpr int H2_NUM_DOFS = 31;

// Canonical deploy/WBC motor-slot order. DDS LowCmd/LowState and MuJoCo sim2sim
// bridge use this order at their boundary.
const std::array<const char*, H2_NUM_DOFS> H2_MOTOR_JOINT_NAMES = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
};

// Canonical policy/IsaacLab order derived from H2_ISAACLAB_TO_MUJOCO_DOF in
// gear_sonic/envs/manager_env/robots/h2.py.
const std::array<const char*, H2_NUM_DOFS> H2_POLICY_JOINT_NAMES = {
    "left_hip_pitch_joint", "left_knee_joint", "right_hip_pitch_joint", "right_knee_joint", "waist_pitch_joint", "left_shoulder_yaw_joint",
    "left_hip_roll_joint", "left_ankle_pitch_joint", "right_hip_roll_joint", "right_ankle_pitch_joint", "head_pitch_joint", "left_elbow_joint",
    "left_hip_yaw_joint", "left_ankle_roll_joint", "right_hip_yaw_joint", "right_ankle_roll_joint", "head_yaw_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "right_shoulder_roll_joint", "right_elbow_joint", "right_wrist_pitch_joint",
    "waist_roll_joint", "left_shoulder_roll_joint", "left_wrist_pitch_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint",
};

constexpr double ONE_DEGREE = 0.0174533;  ///< One degree in radians.

// Motor armature constants (used for PID gain computation)
constexpr double ARMATURE_5020 = 0.003609725;
constexpr double ARMATURE_7520_14 = 0.010177520;
constexpr double ARMATURE_7520_22 = 0.025101925;
constexpr double ARMATURE_4010 = 0.00425;

// Control parameters for PID gain computation
constexpr double NATURAL_FREQ = 10 * 2.0 * 3.1415926535; // 10Hz
constexpr double DAMPING_RATIO = 2;

// Computed stiffness values: stiffness = armature * natural_freq^2
constexpr double STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
constexpr double STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;

// Computed damping values: damping = 2.0 * damping_ratio * armature * natural_freq
constexpr double DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
constexpr double DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
constexpr double DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
constexpr double DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

// Effort limits for different motor types (used for action scale computation)
constexpr double EFFORT_LIMIT_5020 = 25.0;    // 5020 motor type
constexpr double EFFORT_LIMIT_7520_14 = 88.0; // 7520_14 motor type
constexpr double EFFORT_LIMIT_7520_22 = 139.0; // 7520_22 motor type
constexpr double EFFORT_LIMIT_4010 = 5.0;     // 4010 motor type


// ---------------------------------------------------------------------------
// H2 joint ordering helpers
//
// NOTE: Indices below are in IsaacLab DOF order unless otherwise noted.
// These values must match the training-time ordering used by the policy.
// ---------------------------------------------------------------------------

// VR5Point index (MuJoCo body index): left wrist, right wrist, pelvis, left ankle, right ankle
// These indices are consumed by reference-motion helpers and must match the
// exported motion metadata (see reference/*/metadata.txt body_names ordering).
const std::array<int, 5> vr_5point_index = {24, 31, 0, 5, 11};

// VR3Point index (MuJoCo body index): left wrist, right wrist, torso
const std::array<int, 3> vr_3point_index = { 30, 31, 9 };

// Upper body joint index (mujoco order)
const std::vector<int> upper_body_joint_mujoco_order_in_isaaclab_index = {17, 24, 4, 10, 16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30};
const std::vector<int> upper_body_joint_mujoco_order_in_mujoco_index = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

// upper body joint index (isaaclab order)
const std::vector<int> upper_body_joint_isaaclab_order_in_isaaclab_index = {4, 5, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
const std::vector<int> upper_body_joint_isaaclab_order_in_mujoco_index = {14, 19, 15, 20, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30};

// External upper-body interface remains 17 DOF (waist + arms, no H2 head).
// Do not reuse the 19-DOF list above here, otherwise head_pitch/head_yaw shift
// arm targets and read past the 17-element input buffer.
const std::vector<int> upper_body_external_17_isaaclab_order_in_isaaclab_index = {4, 5, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

// wrist joint index (mujoco order)
const std::vector<int> wrist_joint_mujoco_order_in_isaaclab_index = {19, 26, 20, 29, 23, 30};
const std::vector<int> wrist_joint_mujoco_order_in_mujoco_index = {21, 22, 23, 28, 29, 30};

// wrist joint index (isaaclab order)
const std::vector<int> wrist_joint_isaaclab_order_in_isaaclab_index = {19, 20, 23, 26, 29, 30};
const std::vector<int> wrist_joint_isaaclab_order_in_mujoco_index = {21, 23, 29, 22, 28, 30};

// lower body joint index (mujoco order)
const std::vector<int> lower_body_joint_mujoco_order_in_isaaclab_index = {0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15};
const std::vector<int> lower_body_joint_mujoco_order_in_mujoco_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// lower body joint index (isaaclab order)
const std::vector<int> lower_body_joint_isaaclab_order_in_isaaclab_index = {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15};
const std::vector<int> lower_body_joint_isaaclab_order_in_mujoco_index = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};

// Joint mapping arrays.
// Source of truth values: H2_MOTOR_JOINT_NAMES and H2_POLICY_JOINT_NAMES above,
// derived from gear_sonic/envs/manager_env/robots/h2.py.
//
// `H2_MOTOR_TO_POLICY[motor_idx] -> policy_idx`
// `H2_POLICY_TO_MOTOR[policy_idx] -> motor_idx`
const std::array<int, H2_NUM_DOFS> H2_MOTOR_TO_POLICY = {
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10, 16, 18, 25, 5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30
};

const std::array<int, H2_NUM_DOFS> H2_POLICY_TO_MOTOR = {
  0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30
};

// Backward-compatible aliases for existing code. Despite their names, the
// operational meanings are motor->policy and policy->motor respectively.
static const auto& isaaclab_to_mujoco = H2_MOTOR_TO_POLICY;
static const auto& mujoco_to_isaaclab = H2_POLICY_TO_MOTOR;
// Action scaling and PD gains below are synchronized with
// gear_sonic/envs/manager_env/robots/h2.py.
// Formula: Kp = armature * (10*2*pi)^2, Kd = 2*zeta*armature*(10*2*pi), zeta=2.
// Key computed values from h2.py (NATURAL_FREQ=62.83185307, DAMPING_RATIO=2.0):
//   7520_22 hip/roll/pitch/knee: Kp=99.0984277767, Kd=6.3088018535, scale(417Nm)=1.0519843991
//   7520_14 hip_yaw/waist_yaw:   Kp=40.1792384714, Kd=2.5578897650, scale(264Nm)=1.6426393956
//   2x5020 ankle/head/waist_rp:  Kp=28.5012461957, Kd=1.8144456866, scale(150Nm)=1.3157319418
//   5020 arm/wrist_roll:          Kp=14.2506230979, Kd=0.9072228433, scale(75Nm)=1.3157319418
//   4010 wrist_pitch/yaw:         Kp=16.7783274809, Kd=1.0681415022, scale(15Nm)=0.2235026110
static constexpr double H2_STIFFNESS_HIP_YAW = STIFFNESS_7520_14;
static constexpr double H2_STIFFNESS_HIP_ROLL_PITCH_KNEE = STIFFNESS_7520_22;
static constexpr double H2_STIFFNESS_ANKLE_WAIST_RP_HEAD = 2.0 * STIFFNESS_5020;
static constexpr double H2_STIFFNESS_WAIST_YAW = STIFFNESS_7520_14;
static constexpr double H2_STIFFNESS_ARM = STIFFNESS_5020;
static constexpr double H2_STIFFNESS_WRIST_R = STIFFNESS_5020;
static constexpr double H2_STIFFNESS_WRIST_PY = STIFFNESS_4010;

static constexpr double H2_DAMPING_HIP_YAW = DAMPING_7520_14;
static constexpr double H2_DAMPING_HIP_ROLL_PITCH_KNEE = DAMPING_7520_22;
static constexpr double H2_DAMPING_ANKLE_WAIST_RP_HEAD = 2.0 * DAMPING_5020;
static constexpr double H2_DAMPING_WAIST_YAW = DAMPING_7520_14;
static constexpr double H2_DAMPING_ARM = DAMPING_5020;
static constexpr double H2_DAMPING_WRIST_R = DAMPING_5020;
static constexpr double H2_DAMPING_WRIST_PY = DAMPING_4010;

static constexpr double H2_EFFORT_HIP_YAW = 264.0;
static constexpr double H2_EFFORT_HIP_ROLL_PITCH_KNEE = 417.0;
static constexpr double H2_EFFORT_ANKLE_WAIST_RP_HEAD = 150.0;
static constexpr double H2_EFFORT_WAIST_YAW = 264.0;
static constexpr double H2_EFFORT_ARM = 75.0;
static constexpr double H2_EFFORT_WRIST_R = 75.0;
static constexpr double H2_EFFORT_WRIST_PY = 15.0;

const std::array<double, 31> H2_action_scale = {
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // left_hip_pitch_joint
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // left_hip_roll_joint
    0.25 * H2_EFFORT_HIP_YAW / H2_STIFFNESS_HIP_YAW, // left_hip_yaw_joint
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // left_knee_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // left_ankle_pitch_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // left_ankle_roll_joint
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // right_hip_pitch_joint
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // right_hip_roll_joint
    0.25 * H2_EFFORT_HIP_YAW / H2_STIFFNESS_HIP_YAW, // right_hip_yaw_joint
    0.25 * H2_EFFORT_HIP_ROLL_PITCH_KNEE / H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, // right_knee_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // right_ankle_pitch_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // right_ankle_roll_joint
    0.25 * H2_EFFORT_WAIST_YAW / H2_STIFFNESS_WAIST_YAW, // waist_yaw_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // waist_roll_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // waist_pitch_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // head_pitch_joint
    0.25 * H2_EFFORT_ANKLE_WAIST_RP_HEAD / H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, // head_yaw_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // left_shoulder_pitch_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // left_shoulder_roll_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // left_shoulder_yaw_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // left_elbow_joint
    0.25 * H2_EFFORT_WRIST_R / H2_STIFFNESS_WRIST_R, // left_wrist_roll_joint
    0.25 * H2_EFFORT_WRIST_PY / H2_STIFFNESS_WRIST_PY, // left_wrist_pitch_joint
    0.25 * H2_EFFORT_WRIST_PY / H2_STIFFNESS_WRIST_PY, // left_wrist_yaw_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // right_shoulder_pitch_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // right_shoulder_roll_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // right_shoulder_yaw_joint
    0.25 * H2_EFFORT_ARM / H2_STIFFNESS_ARM, // right_elbow_joint
    0.25 * H2_EFFORT_WRIST_R / H2_STIFFNESS_WRIST_R, // right_wrist_roll_joint
    0.25 * H2_EFFORT_WRIST_PY / H2_STIFFNESS_WRIST_PY, // right_wrist_pitch_joint
    0.25 * H2_EFFORT_WRIST_PY / H2_STIFFNESS_WRIST_PY, // right_wrist_yaw_joint
};

// PID control gains - Position gains (Kp)
// Source of truth: gear_sonic/envs/manager_env/robots/h2.py actuator stiffness.
// Order must match H2_MOTOR_JOINT_NAMES (motor slot order).
const std::array<double, 31> kps = {
    H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_HIP_YAW, H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD,
    H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_HIP_YAW, H2_STIFFNESS_HIP_ROLL_PITCH_KNEE, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD,
    H2_STIFFNESS_WAIST_YAW, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD,
    H2_STIFFNESS_ANKLE_WAIST_RP_HEAD, H2_STIFFNESS_ANKLE_WAIST_RP_HEAD,
    H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_WRIST_R, H2_STIFFNESS_WRIST_PY, H2_STIFFNESS_WRIST_PY,
    H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_ARM, H2_STIFFNESS_WRIST_R, H2_STIFFNESS_WRIST_PY, H2_STIFFNESS_WRIST_PY,
};

// PID control gains - Derivative gains (Kd)
// Source of truth: gear_sonic/envs/manager_env/robots/h2.py actuator damping.
// Order must match H2_MOTOR_JOINT_NAMES (motor slot order).
const std::array<double, 31> kds = {
    H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_HIP_YAW, H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_ANKLE_WAIST_RP_HEAD, H2_DAMPING_ANKLE_WAIST_RP_HEAD,
    H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_HIP_YAW, H2_DAMPING_HIP_ROLL_PITCH_KNEE, H2_DAMPING_ANKLE_WAIST_RP_HEAD, H2_DAMPING_ANKLE_WAIST_RP_HEAD,
    H2_DAMPING_WAIST_YAW, H2_DAMPING_ANKLE_WAIST_RP_HEAD, H2_DAMPING_ANKLE_WAIST_RP_HEAD,
    H2_DAMPING_ANKLE_WAIST_RP_HEAD, H2_DAMPING_ANKLE_WAIST_RP_HEAD,
    H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_WRIST_R, H2_DAMPING_WRIST_PY, H2_DAMPING_WRIST_PY,
    H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_ARM, H2_DAMPING_WRIST_R, H2_DAMPING_WRIST_PY, H2_DAMPING_WRIST_PY,
};

// Default joint angles synchronized with H2_CFG.init_state.joint_pos in h2.py.
const std::array<double, 31> default_angles = {
    -0.312f, // left_hip_pitch_joint
    0.0f, // left_hip_roll_joint
    0.0f, // left_hip_yaw_joint
    0.669f, // left_knee_joint
    -0.363f, // left_ankle_pitch_joint
    0.0f, // left_ankle_roll_joint
    -0.312f, // right_hip_pitch_joint
    0.0f, // right_hip_roll_joint
    0.0f, // right_hip_yaw_joint
    0.669f, // right_knee_joint
    -0.363f, // right_ankle_pitch_joint
    0.0f, // right_ankle_roll_joint
    0.0f, // waist_yaw_joint
    0.0f, // waist_roll_joint
    0.0f, // waist_pitch_joint
    0.0f, // head_pitch_joint
    0.0f, // head_yaw_joint
    0.2f, // left_shoulder_pitch_joint
    0.2f, // left_shoulder_roll_joint
    0.0f, // left_shoulder_yaw_joint
    0.6f, // left_elbow_joint
    0.0f, // left_wrist_roll_joint
    0.0f, // left_wrist_pitch_joint
    0.0f, // left_wrist_yaw_joint
    0.2f, // right_shoulder_pitch_joint
    -0.2f, // right_shoulder_roll_joint
    0.0f, // right_shoulder_yaw_joint
    0.6f, // right_elbow_joint
    0.0f, // right_wrist_roll_joint
    0.0f, // right_wrist_pitch_joint
    0.0f // right_wrist_yaw_joint
};

#endif // POLICY_PARAMETERS_HPP
