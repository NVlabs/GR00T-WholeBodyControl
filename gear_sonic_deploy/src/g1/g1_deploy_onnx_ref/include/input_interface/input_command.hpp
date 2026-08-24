/**
 * @file input_command.hpp
 * @brief Lightweight message structs used for inter-component communication
 *        between ZMQ subscribers and the input managers.
 *
 * Two message types are defined:
 *  - CommandMessage  – carries high-level control signals (start / stop /
 *                      planner-mode toggle) received on the ZMQ "command" topic.
 *  - PlannerMessage  – carries per-frame locomotion commands (mode, movement
 *                      direction, facing direction, speed, height, and optional
 *                      upper-body / hand data) received on the ZMQ "planner" topic.
 *
 * Both structs are plain-old-data (POD-like) value types designed to be written
 * under a mutex by a background ZMQ thread and read by the main control loop.
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>

#include "control_session_state.hpp"
#include "../localmotion_kplanner.hpp"  // For LocomotionMode enum

// ---------------------------------------------------------------------------
// CommandMessage
// ---------------------------------------------------------------------------
/**
 * @brief Wire format for the ZMQ "command" topic.
 *
 * Packed-message protocol v2 fields sent by the remote controller:
 *   - receiver_epoch: u8[16] - current native deployment challenge
 *   - publisher_session: u8[16] - nonzero publisher-process identifier
 *   - command_index: i64[1] - nonnegative replay sequence; an identical
 *     claim retry may repeat, while accepted control actions strictly advance
 *   - claim: bool[1] - claim deployment; mutually exclusive with start/stop
 *   - start: bool[1] - start selected mode; mutually exclusive with stop/claim
 *   - stop: bool[1] - terminal stop; mutually exclusive with start/claim
 *   - planner: bool[1] - select planner or streamed-motion mode
 *
 * The manager must claim the receiver epoch before start. It permanently
 * binds to the first publisher session for process lifetime; stop, deadman,
 * mode changes, and disconnects never reset binding or replay sequences.
 */
struct CommandMessage {
  ControlSessionToken receiver_epoch{};    ///< Native process challenge.
  ControlSessionToken publisher_session{}; ///< Publisher process identifier.
  int64_t command_index = -1;              ///< Monotonic command replay sequence.
  bool claim = false;                      ///< Request permanent ownership.
  bool start = false;     ///< When true, request the control system to start.
  bool stop = false;      ///< When true, request an emergency / graceful stop.
  bool planner = false;   ///< true  → planner mode  (use planner topic for locomotion)
                          ///< false → streamed-motion mode  (use pose topic)
  /// Optional absolute heading override (radians).  When set, the value is
  /// written directly into HeadingState.delta_heading.
  std::optional<double> delta_heading;
  bool valid = false;     ///< Set to true once a message has been decoded successfully.
};

// ---------------------------------------------------------------------------
// PlannerMessage
// ---------------------------------------------------------------------------
/**
 * @brief Wire format for the ZMQ "planner" topic.
 *
 * Required fields (must be present in every protocol-v2 message):
 *   - receiver_epoch: u8[16] - current native deployment challenge
 *   - publisher_session: u8[16] - permanently claimed publisher identifier
 *   - frame_index: int64 – strictly increasing publisher sequence
 *   - mode      : int32  – LocomotionMode enum cast (IDLE, WALK, RUN, …)
 *   - movement  : float[3] – desired movement direction unit vector (x, y, z)
 *   - facing    : float[3] – desired facing direction unit vector  (x, y, z)
 *
 * Optional fields (may or may not be present):
 *   - speed              : float – desired locomotion speed (-1.0 = use default)
 *   - height             : float – desired body height      (-1.0 = use default)
 *   - upper_body_position: float[17] – target upper-body joint positions  (radians)
 *   - upper_body_velocity: float[17] – target upper-body joint velocities (rad/s)
 *   - left_hand_joints   : float[7]  – Dex3 left-hand joint positions
 *   - right_hand_joints  : float[7]  – Dex3 right-hand joint positions
 *
 * The `timestamp` field is set only after full validation and backs the
 * manager's 500 ms active-input lease.
 */
struct PlannerMessage {
  bool valid = false;  ///< True once this struct contains a successfully decoded message.

  ControlSessionToken receiver_epoch{};
  ControlSessionToken publisher_session{};

  /// Strictly increasing publisher sequence used to reject replayed packets.
  int64_t frame_index = -1;

  /// Locomotion mode (cast of LocomotionMode enum). Defaults to IDLE.
  int mode = static_cast<int>(LocomotionMode::IDLE);

  /// Desired movement direction as a 3D unit vector [x, y, z].
  /// Zeroed when the robot should stand still.
  std::array<double, 3> movement = {0.0, 0.0, 0.0};

  /// Desired facing direction as a 3D unit vector [x, y, z].
  /// Defaults to facing forward along the +X axis.
  std::array<double, 3> facing = {1.0, 0.0, 0.0};

  /// Optional upper-body joint target positions (17 DOF, radians).
  /// Present when the remote controller provides whole-body commands.
  std::optional<std::array<double, 17>> upper_body_position;

  /// Optional upper-body joint target velocities (17 DOF, rad/s).
  std::optional<std::array<double, 17>> upper_body_velocity;

  /// Optional left-hand Dex3 joint positions (7 DOF).
  std::optional<std::array<double, 7>> left_hand_joints;

  /// Optional right-hand Dex3 joint positions (7 DOF).
  std::optional<std::array<double, 7>> right_hand_joints;

  /// Optional VR 3-point positions (left wrist, right wrist, head).
  std::optional<std::array<double, 9>> vr_position;

  /// Optional VR 3-point orientations (three quaternions).
  std::optional<std::array<double, 12>> vr_orientation;

  /// Optional VR 3-point compliance values.
  std::optional<std::array<double, 3>> vr_compliance;

  /// Desired locomotion speed.  -1.0 means "use the default for the current mode".
  double speed = -1.0;

  /// Desired body height.  -1.0 means "use the default for the current mode".
  double height = -1.0;

  /// Local steady-clock timestamp recorded after full message validation.
  /// Used by the manager's active-input lease.
  std::chrono::steady_clock::time_point timestamp{};
};

