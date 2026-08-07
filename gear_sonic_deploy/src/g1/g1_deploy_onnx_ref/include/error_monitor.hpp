/**
 * @file error_monitor.hpp
 * @brief Monitors motor error states from LowState and prints warnings on
 *        state transitions, with periodic reminders for persistent faults.
 *
 * The G1 robot reports per-motor fault codes via the `motorstate()` field in
 * each MotorState_ message (part of LowState_ on topic `rt/lowstate`).  A
 * non-zero value indicates a motor fault.
 *
 * ErrorMonitor tracks the previous motorstate value for each motor and:
 *  1. Prints immediately on state transitions (fault appears / clears / changes).
 *  2. Prints a periodic summary while any faults persist, so the message
 *     is not lost in other log output.
 *
 * Called at 500 Hz from LowStateHandler.  The reminder interval defaults to
 * every 2500 calls (~5 seconds).
 */

#pragma once

#include <array>
#include <cstdint>
#include <iostream>

#include "robot_parameters.hpp"

/**
 * @class ErrorMonitor
 * @brief Detects motor fault transitions and periodically reminds about
 *        persistent faults.
 *
 * Usage: call `update()` with the current motor error codes from each
 * LowState callback.
 */
class ErrorMonitor {
 public:
  /**
   * @param reminder_interval  Number of update() calls between periodic
   *        reminders while faults persist.  At 500 Hz, 2500 = every ~5 s.
   */
  explicit ErrorMonitor(int reminder_interval = 500)
    : reminder_interval_(reminder_interval) {
    prev_motorstate_.fill(0);
  }

  /**
   * @brief Check motor error states, print on transitions and periodic reminders.
   * @param motorstates Array of motorstate error codes, one per motor.
   */
  void update(const std::array<uint32_t, G1_NUM_MOTOR>& motorstates) {
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      uint32_t state = motorstates[i];
      if (state != prev_motorstate_[i]) {
        if (state != 0 && prev_motorstate_[i] == 0) {
          // New fault
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i)
                    << ") FAULT: code 0x" << std::hex << state << std::dec
                    << std::endl;
          error_count_++;
        } else if (state == 0 && prev_motorstate_[i] != 0) {
          // Fault cleared
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i)
                    << ") fault CLEARED" << std::endl;
          if (error_count_ > 0) error_count_--;
        } else {
          // Fault code changed
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i)
                    << ") fault CHANGED: 0x" << std::hex << prev_motorstate_[i]
                    << " -> 0x" << state << std::dec << std::endl;
        }
        prev_motorstate_[i] = state;
        reminder_counter_ = 0;  // reset reminder timer on any transition
      }
    }

    // Periodic reminder while faults persist
    if (error_count_ > 0) {
      reminder_counter_++;
      if (reminder_counter_ >= reminder_interval_) {
        reminder_counter_ = 0;
        std::cout << "[ErrorMonitor] " << error_count_
                  << " motor(s) still faulted:";
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          if (prev_motorstate_[i] != 0) {
            std::cout << " " << jointName(i) << "(0x" << std::hex
                      << prev_motorstate_[i] << std::dec << ")";
          }
        }
        std::cout << std::endl;
      }
    }
  }

  /// Number of motors currently in a faulted state.
  int getErrorCount() const { return error_count_; }

  /// True if any motor is currently faulted.
  bool hasErrors() const { return error_count_ > 0; }

 private:
  std::array<uint32_t, G1_NUM_MOTOR> prev_motorstate_;
  int error_count_ = 0;
  int reminder_interval_ = 2500;
  int reminder_counter_ = 0;

  static const char* jointName(int index) {
    static const char* names[] = {
      "LeftHipPitch",       // 0
      "LeftHipRoll",        // 1
      "LeftHipYaw",         // 2
      "LeftKnee",           // 3
      "LeftAnklePitch",     // 4
      "LeftAnkleRoll",      // 5
      "RightHipPitch",      // 6
      "RightHipRoll",       // 7
      "RightHipYaw",        // 8
      "RightKnee",          // 9
      "RightAnklePitch",    // 10
      "RightAnkleRoll",     // 11
      "WaistYaw",           // 12
      "WaistRoll",          // 13
      "WaistPitch",         // 14
      "LeftShoulderPitch",  // 15
      "LeftShoulderRoll",   // 16
      "LeftShoulderYaw",    // 17
      "LeftElbow",          // 18
      "LeftWristRoll",      // 19
      "LeftWristPitch",     // 20
      "LeftWristYaw",       // 21
      "RightShoulderPitch", // 22
      "RightShoulderRoll",  // 23
      "RightShoulderYaw",   // 24
      "RightElbow",         // 25
      "RightWristRoll",     // 26
      "RightWristPitch",    // 27
      "RightWristYaw",      // 28
    };
    if (index >= 0 && index < G1_NUM_MOTOR) return names[index];
    return "Unknown";
  }
};
