/**
 * @file robot_parameters.hpp
 * @brief Hardware-level constants, data structures, and joint indices for the
 *        Unitree H2 humanoid robot.
 *
 * This header defines:
 *  - DDS topic names for the Unitree SDK (low-level command / state channels).
 *  - Motor count and per-joint MotorCommand structure.
 *  - HeadingState – a compact struct bundling the captured IMU quaternion
 *    with a user-controlled delta-heading offset.
 *  - OperatorState – high-level start / stop / play flags.
 *  - Control mode enum (series vs. parallel ankle actuation).
 *  - H2JointIndex enum mapping symbolic joint names to hardware motor indices.
 *
 * All indices in this file use the **hardware / URDF ordering** (not the
 * IsaacLab training ordering – see `policy_parameters.hpp` for the mapping).
 */

#ifndef ROBOT_PARAMETERS_HPP
#define ROBOT_PARAMETERS_HPP

#include <array>

// ---------------------------------------------------------------------------
// Unitree SDK DDS topic names
// ---------------------------------------------------------------------------
static const std::string HG_CMD_TOPIC = "rt/lowcmd";       ///< Low-level motor command topic.
static const std::string HG_IMU_TORSO = "rt/secondary_imu";///< Secondary (torso) IMU topic.
static const std::string HG_STATE_TOPIC = "rt/lowstate";    ///< Low-level motor / sensor state topic.

/// Total number of actuated joints on the H2 (31-DOF configuration).
constexpr size_t H2_NUM_MOTOR = 31;

/**
 * @brief Per-joint motor command sent to the low-level controller.
 *
 * Each field is an array of size H2_NUM_MOTOR (31), indexed by hardware joint
 * index (see H2JointIndex).
 */
struct MotorCommand {
    std::array<float, H2_NUM_MOTOR> q_target = {};   ///< Target position (rad).
    std::array<float, H2_NUM_MOTOR> dq_target = {};  ///< Target velocity (rad/s).
    std::array<float, H2_NUM_MOTOR> kp = {};          ///< Position gain (Nm/rad).
    std::array<float, H2_NUM_MOTOR> kd = {};          ///< Velocity gain (Nms/rad).
    std::array<float, H2_NUM_MOTOR> tau_ff = {};      ///< Feed-forward torque (Nm).
};

/**
 * @brief Bundled heading state for thread-safe access via DataBuffer.
 *
 * Captures both the initial IMU base quaternion (set when heading is
 * reinitialised) and a user-adjustable delta heading offset (adjusted via
 * keyboard Q/E or D-pad).
 */
struct HeadingState {
    std::array<double, 4> init_base_quat;  ///< Captured IMU base quaternion (w,x,y,z) at init.
    double delta_heading;                   ///< Cumulative heading offset (radians).
    
    HeadingState(const std::array<double, 4>& quat = {1.0, 0.0, 0.0, 0.0}, double delta = 0.0)
        : init_base_quat(quat), delta_heading(delta) {}
};

/**
 * @brief High-level operator signals (set by input interfaces, read by control loop).
 */
struct OperatorState {
  bool stop = false;   ///< Emergency stop requested.
  bool start = false;  ///< Control-system start requested.
  bool play = false;   ///< Motion playback active.
};

/**
 * @brief Ankle actuation mode.
 *
 * The G1's ankle uses a coupled 2-DOF mechanism that can be controlled
 * in series (pitch/roll) or parallel (A/B motor) modes.
 */
enum class Mode {
  PR = 0, ///< Series control for Pitch / Roll joints.
  AB = 1  ///< Parallel control for A / B motors.
};

/**
 * @brief Symbolic names for G1 hardware joint indices (0–28).
 *
 * Ankle joints have dual names because they can be addressed in either
 * series (Pitch/Roll) or parallel (A/B) mode.  Joints marked "INVALID"
 * are not present on the 23-DOF or waist-locked 29-DOF variants.
 */
enum H2JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,
  WaistA = 13,
  WaistPitch = 14,
  WaistB = 14,
  HeadPitch = 15,
  HeadYaw = 16,
  LeftShoulderPitch = 17,
  LeftShoulderRoll = 18,
  LeftShoulderYaw = 19,
  LeftElbow = 20,
  LeftWristRoll = 21,
  LeftWristPitch = 22,
  LeftWristYaw = 23,
  RightShoulderPitch = 24,
  RightShoulderRoll = 25,
  RightShoulderYaw = 26,
  RightElbow = 27,
  RightWristRoll = 28,
  RightWristPitch = 29,
  RightWristYaw = 30
};

#endif // ROBOT_PARAMETERS_HPP
