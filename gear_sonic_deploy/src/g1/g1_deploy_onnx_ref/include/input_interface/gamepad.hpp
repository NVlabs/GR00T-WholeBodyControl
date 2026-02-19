/**
 * @file gamepad.hpp
 * @brief Unitree wireless gamepad driver and InputInterface implementation.
 *
 * This file contains:
 *  1. Low-level data structures that map directly onto the raw 40-byte
 *     wireless-remote packet received from the Unitree SDK
 *     (xKeySwitchUnion, xRockerBtnDataStruct, REMOTE_DATA_RX).
 *  2. A simple edge-detecting Button helper class.
 *  3. The Gamepad class – a full InputInterface that translates raw button
 *     and analog-stick data into motion / planner commands.
 *
 * The Gamepad supports two operational modes:
 *  - **Non-planner mode** – uses the face buttons and bumpers to cycle through
 *    pre-loaded reference motions and play/pause/restart them.
 *  - **Planner mode** (toggled with F1) – the left stick controls movement
 *    direction, the right stick controls facing direction, and the bumpers
 *    switch between locomotion modes (idle / slow walk / walk / run / squat /
 *    kneel).  L2/R2 adjust speed or height depending on the current mode.
 *
 * See the button-mapping comment block inside the class for the complete mapping.
 */

#ifndef GAMEPAD_HPP
#define GAMEPAD_HPP

#include <cmath>
#include <iostream>
#include <array>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "input_interface.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace unitree::common {

// =========================================================================
// Raw wireless-remote data structures (match the Unitree SDK wire format)
// =========================================================================

/**
 * @brief Bit-field union for the 16-bit button state word.
 *
 * Each member is a single-bit flag corresponding to one physical button.
 * The `value` member gives access to all 16 bits as a single uint16_t.
 */
typedef union {
    struct {
        uint8_t R1 : 1;      ///< Right bumper
        uint8_t L1 : 1;      ///< Left bumper
        uint8_t start : 1;   ///< Start button
        uint8_t select : 1;  ///< Select / back button
        uint8_t R2 : 1;      ///< Right trigger (digital)
        uint8_t L2 : 1;      ///< Left trigger (digital)
        uint8_t F1 : 1;      ///< Function 1 button
        uint8_t F2 : 1;      ///< Function 2 button
        uint8_t A : 1;       ///< A face button
        uint8_t B : 1;       ///< B face button
        uint8_t X : 1;       ///< X face button
        uint8_t Y : 1;       ///< Y face button
        uint8_t up : 1;      ///< D-pad up
        uint8_t right : 1;   ///< D-pad right
        uint8_t down : 1;    ///< D-pad down
        uint8_t left : 1;    ///< D-pad left
    } components;

    uint16_t value;  ///< Raw 16-bit button word.
} xKeySwitchUnion;

/**
 * @brief 40-byte raw joystick packet (24 bytes used, 16 idle/reserved).
 *
 * Layout matches the Unitree wireless remote receiver format.
 */
typedef struct {
    uint8_t head[2];        ///< 2-byte packet header
    xKeySwitchUnion btn;    ///< 16-bit button state
    float lx;               ///< Left stick horizontal  (−1..+1)
    float rx;               ///< Right stick horizontal  (−1..+1)
    float ry;               ///< Right stick vertical    (−1..+1)
    float L2;               ///< Left trigger analog     (0..+1)
    float ly;               ///< Left stick vertical     (−1..+1)

    uint8_t idle[16];       ///< Reserved / unused bytes
} xRockerBtnDataStruct;

/**
 * @brief Union overlay giving byte-array access to the full 40-byte packet.
 *
 * `RF_RX` provides structured access; `buff` gives raw byte-level access
 * (useful for memcpy from the SDK callback).
 */
typedef union {
    xRockerBtnDataStruct RF_RX;
    uint8_t buff[40] = {0};
} REMOTE_DATA_RX;

/**
 * @brief Simple edge-detecting button helper.
 *
 * Call update() once per frame with the current digital state.
 * After the call:
 *   - `pressed`    – true while the button is held down.
 *   - `on_press`   – true only on the frame the button transitions from released → pressed.
 *   - `on_release` – true only on the frame the button transitions from pressed → released.
 */
class Button {
  public:
    Button() {}

    /// Update the button state for this frame.
    void update(bool state) {
      on_press = state ? state != pressed : false;
      on_release = state ? false : state != pressed;
      pressed = state;
    }

    bool pressed = false;     ///< Currently held down.
    bool on_press = false;    ///< Rising edge this frame.
    bool on_release = false;  ///< Falling edge this frame.
};

/**
 * @class Gamepad
 * @brief InputInterface implementation for the Unitree wireless gamepad.
 *
 * Reads raw joystick packets via `gamepad_data` (populated externally by
 * InterfaceManager::UpdateGamepadRemoteData) and translates button presses /
 * stick deflections into per-frame action flags.
 *
 * Supports two modes:
 *  - Reference-motion mode (default): L1/R1 cycle motions, A plays, B restarts.
 *  - Planner mode (toggled by F1):    sticks control direction / facing,
 *    L1/R1 change locomotion mode, L2/R2 adjust speed/height.
 */
class Gamepad : public InputInterface {
  public:
    /// Compile-time toggle for debug log output.  Set to false and rebuild to suppress.
    static constexpr bool DEBUG_LOGGING = false;
    
    // ------------------------------------------------------------------
    // Per-frame action flags (reset at the start of every update() call)
    // ------------------------------------------------------------------
    bool motion_prev = false;      ///< Switch to previous pre-loaded motion.
    bool motion_next = false;      ///< Switch to next pre-loaded motion.
    bool play_motion = false;      ///< Start / resume motion playback.
    bool motion_restart = false;   ///< Reset current motion to frame 0 (paused).

    bool start_control = false;    ///< Request control-system start.
    bool stop_control = false;     ///< Request emergency stop.

    bool delta_left = false;       ///< Nudge heading left  (+0.1 rad).
    bool delta_right = false;      ///< Nudge heading right (−0.1 rad).

    bool reinitialize = false;     ///< Recapture the IMU base quaternion.

    bool use_planner = false;      ///< True when planner mode is active (toggle via F1).

    // ------------------------------------------------------------------
    // Planner-mode state (persists across frames)
    // ------------------------------------------------------------------
    int planner_use_movement_mode = 1;         ///< Locomotion mode index (1=slow walk, 2=walk, 3=run, …).
    double planner_use_movement_speed = -1;    ///< Desired speed (−1 = mode default).
    double planner_use_height = -1;            ///< Desired body height (−1 = mode default).
    bool planner_emergency_stop = false;       ///< Immediate halt flag (B button in planner mode).
    double planner_facing_angle = 0.0;         ///< Accumulated facing angle from right stick (radians).
    double planner_moving_direction = 0.0;     ///< Current movement direction from left stick (radians).

    /// Raw wireless-remote data buffer.  Written externally (e.g. by InterfaceManager).
    REMOTE_DATA_RX gamepad_data;

    explicit Gamepad() : InputInterface() {
      type_ = InputType::GAMEPAD;
    }

    /*
     * Gamepad Button Mappings:
     * 
     * Common (both modes):
     * - A Button      - Play motion
     * - X/Y Buttons   - Reinitialize base quaternion and delta heading
     * - Start Button  - Start control
     * - Select Button - Emergency Stop (kills all motion)
     * - F1 Button - Toggle planner on/off
     * - D-pad L/R     - Delta heading left/right (+/-0.1 rad)
     * 
     * Non-Planner Mode:
     * - B Button      - Restart current motion and pause (emergency stop)
     * - L1/R1 Buttons - Previous/Next motion
     * 
     * Planner Mode:
     * - B Button      - Emergency stop
     * - L1/R1 Buttons - Change movement mode (0-4: idle, slow walk, walk, run, boxing)
     * - L2/R2 Buttons - Decrease/Increase movement speed
     * - Left Stick    - Movement direction (lx, ly)
     * - Right Stick   - Facing direction (rx, ry)
     */

    // Flag to trigger safety reset in handle_input
    bool trigger_safety_reset = false;

    // Override the update function from InputInterface
    void update() override {
      // Check for safety reset trigger from manager
      if (CheckAndClearSafetyReset()) {
        use_planner = false;
        trigger_safety_reset = true;
        std::cout << "[Gamepad] Safety reset triggered: will disable planner and return to reference motion" << std::endl;
      }

      // Reset input flags each frame
      start_control = false;
      stop_control = false;
      motion_prev = false;
      motion_next = false;
      play_motion = false;
      motion_restart = false;
      delta_left = false;
      delta_right = false;
      reinitialize = false;
      planner_emergency_stop = false;

      // Process gamepad input and set flags based on current button states
      update_gamepad_data(gamepad_data.RF_RX);

      // Debug: Log analog stick values if they're above dead zone
      if constexpr (DEBUG_LOGGING) {
        if (std::abs(lx) > dead_zone || std::abs(ly) > dead_zone) {
          std::cout << "[GAMEPAD DEBUG] Left stick: lx=" << lx << ", ly=" << ly << std::endl;
        }
        if (std::abs(rx) > dead_zone || std::abs(ry) > dead_zone) {
          std::cout << "[GAMEPAD DEBUG] Right stick: rx=" << rx << ", ry=" << ry << std::endl;
        }
      }

      // Button mappings:

      if (!use_planner) {
        // Y - Reinitialize
        if (Y.on_press || X.on_press) { 
          reinitialize = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] X/Y pressed - Reinitialize" << std::endl;
          }
        }
        // B - Reset motion
        if (B.on_press) { 
          motion_restart = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] B pressed - Motion restart" << std::endl;
          }
        }
        // A - Play/Pause motion
        if (A.on_press) { 
          play_motion = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] A pressed - Play/Pause motion" << std::endl;
          }
        }
        // L1/R1 -Motion prev/next
        if (L1.on_press) { 
          motion_prev = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] L1 pressed - Previous motion" << std::endl;
          }
        }
        if (R1.on_press) { 
          motion_next = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] R1 pressed - Next motion" << std::endl;
          }
        }
      }
      else {
        // Y - Reinitialize
        if (Y.on_press || X.on_press) { 
          reinitialize = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] X/Y pressed - Reinitialize" << std::endl;
          }
        }
        // A - Play/Pause motion
        if (A.on_press) { 
          play_motion = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] A pressed - Play/Pause motion" << std::endl;
          }
        }
        // B - Reset motion
        if (B.on_press) { 
          planner_emergency_stop = true; 
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] B pressed - Planner emergency stop" << std::endl;
          }
        }
        // L1/R1 - change movement mode
        if (L1.on_press) { 
          planner_use_movement_mode -= 1; 
          if (planner_use_movement_mode < 0) { planner_use_movement_mode = 6; }
          // If movement mode is 6, i.e. switch from idle to kneel, set height to 0.8
          if (planner_use_movement_mode == 6) { planner_use_height = 0.8;}
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] L1 pressed - Movement mode changed to: " << planner_use_movement_mode << std::endl;
          }
        }
        if (R1.on_press) { 
          planner_use_movement_mode += 1; 
          if (planner_use_movement_mode > 6) { planner_use_movement_mode = 0; }
          // If movement mode is 4, i.e. switch from run to squat, set height to 0.8
          if (planner_use_movement_mode == 4) { planner_use_height = 0.8;}
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GAMEPAD DEBUG] R1 pressed - Movement mode changed to: " << planner_use_movement_mode << std::endl;
          }
        }
        // L2/R2 - change movement speed
        if (R2.pressed) { 
          if (planner_use_movement_mode < 4) {
            planner_use_movement_speed += 0.02;
            if constexpr (DEBUG_LOGGING) {
              if (R2.pressed) std::cout << "[GAMEPAD DEBUG] R2 pressed - Speed increasing" << "Current speed: " << planner_use_movement_speed << std::endl;
            }
          } else {
            planner_use_height += 0.02;
            if constexpr (DEBUG_LOGGING) {
              if (R2.pressed) std::cout << "[GAMEPAD DEBUG] R2 pressed - Height increasing" << "Current height: " << planner_use_height << std::endl;
            }
          }

        }
        if (L2.pressed) { 
          if (planner_use_movement_mode < 4) {
            planner_use_movement_speed -= 0.02;
            if constexpr (DEBUG_LOGGING) {
              if (L2.pressed) std::cout << "[GAMEPAD DEBUG] L2 pressed - Speed decreasing" << "Current speed: " << planner_use_movement_speed << std::endl;
            }
          } else {
            planner_use_height -= 0.02;
            if constexpr (DEBUG_LOGGING) {
              if (L2.pressed) std::cout << "[GAMEPAD DEBUG] L2 pressed - Height decreasing" << "Current height: " << planner_use_height << std::endl;
            }
          }

        }
        // Limit movement speed and height to the range of the movement mode
        switch (planner_use_movement_mode) {
          case 0:
            planner_use_movement_speed = -1.0;
            planner_use_height = -1.0;
            break;
          case 1: // slow walk: 0.1 - 0.8
            planner_use_movement_speed = std::max(planner_use_movement_speed, 0.2);
            planner_use_movement_speed = std::min(planner_use_movement_speed, 0.8);
            planner_use_height = -1.0;
            break;
          case 2: // walk: 0.8 - 2.5
            planner_use_movement_speed = std::max(planner_use_movement_speed, 0.8);
            planner_use_movement_speed = std::min(planner_use_movement_speed, 1.5);
            planner_use_height = -1.0;
            break;
          case 3: // run: 2.5 - 7.5
            planner_use_movement_speed = std::max(planner_use_movement_speed, 1.5);
            planner_use_movement_speed = std::min(planner_use_movement_speed, 3.0);
            planner_use_height = -1.0;
            break;
          case 4: // squat: 
            planner_use_movement_speed = -1.0;
            planner_use_height = std::max(planner_use_height, 0.1);
            planner_use_height = std::min(planner_use_height, 0.8);
            break;
          case 5: // kneel two legs: 
            planner_use_movement_speed = -1.0;
            planner_use_height = std::max(planner_use_height, 0.1);
            planner_use_height = std::min(planner_use_height, 0.8);
            break;
          case 6: // kneel:
            planner_use_movement_speed = -1.0;
            planner_use_height = std::max(planner_use_height, 0.1);
            planner_use_height = std::min(planner_use_height, 0.8);
            break;
        }
        // Analog sticks - change movement and facing direction (with dead zone)
        
        if (std::abs(rx) > dead_zone || std::abs(ry) > dead_zone) {
            planner_facing_angle = planner_facing_angle - 0.02 * rx;
            if constexpr (DEBUG_LOGGING) {
              {
                std::cout << "[GAMEPAD DEBUG] Right stick - Facing angle: " << planner_facing_angle << " rad ("
                          << (planner_facing_angle * 180.0 / M_PI) << " deg)" << std::endl;
              }
            }
        }
        
        if (std::abs(lx) > dead_zone || std::abs(ly) > dead_zone) {
            planner_moving_direction = atan2(ly, lx) - M_PI/2 + planner_facing_angle;
            if constexpr (DEBUG_LOGGING) {
              {
                std::cout << "[GAMEPAD DEBUG] Left stick - Moving direction: " << planner_moving_direction << " rad ("
                          << (planner_moving_direction * 180.0 / M_PI) << " deg)" << std::endl;
              }
            }
        }
        
        // Log speed changes after limit processing
        if constexpr (DEBUG_LOGGING) {
          static double prev_speed = planner_use_movement_speed;
          if (std::abs(planner_use_movement_speed - prev_speed) > 0.01) {
            std::cout << "[GAMEPAD DEBUG] Movement speed: " << planner_use_movement_speed << std::endl;
            prev_speed = planner_use_movement_speed;
          }
        }
      }


      // F1 - Toggle planner
      if (F1.on_press) { 
        use_planner = !use_planner; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GAMEPAD DEBUG] F1 pressed - Planner toggled to: " << (use_planner ? "ON" : "OFF") << std::endl;
        }
      }

      // start - Start control
      if (start.on_press) { 
        start_control = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GAMEPAD DEBUG] Start pressed - Start control" << std::endl;
        }
      }
      // select - Emergency stop
      if (select.on_press) { 
        stop_control = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GAMEPAD DEBUG] Select pressed - Emergency Stop" << std::endl;
        }
      }       
      // left/right - Delta heading left/right
      if (left.on_press) { 
        delta_left = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GAMEPAD DEBUG] D-pad Left pressed - Delta heading left" << std::endl;
        }
      }
      if (right.on_press) { 
        delta_right = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GAMEPAD DEBUG] D-pad Right pressed - Delta heading right" << std::endl;
        }
      }
    }

    /**
     * @brief Decode raw joystick packet into smoothed analog values and Button states.
     *
     * Analog values are smoothed with an exponential moving average (controlled
     * by `smooth`) and dead-zone filtered.  Each Button is updated with edge
     * detection so callers can distinguish press / release / held.
     */
    void update_gamepad_data(xRockerBtnDataStruct& key_data) {
      lx = lx * (1 - smooth) + (std::fabs(key_data.lx) < dead_zone ? 0.0 : key_data.lx) * smooth;
      rx = rx * (1 - smooth) + (std::fabs(key_data.rx) < dead_zone ? 0.0 : key_data.rx) * smooth;
      ry = ry * (1 - smooth) + (std::fabs(key_data.ry) < dead_zone ? 0.0 : key_data.ry) * smooth;
      l2 = l2 * (1 - smooth) + (std::fabs(key_data.L2) < dead_zone ? 0.0 : key_data.L2) * smooth;
      ly = ly * (1 - smooth) + (std::fabs(key_data.ly) < dead_zone ? 0.0 : key_data.ly) * smooth;

      R1.update(key_data.btn.components.R1);
      L1.update(key_data.btn.components.L1);
      start.update(key_data.btn.components.start);
      select.update(key_data.btn.components.select);
      R2.update(key_data.btn.components.R2);
      L2.update(key_data.btn.components.L2);
      F1.update(key_data.btn.components.F1);
      F2.update(key_data.btn.components.F2);
      A.update(key_data.btn.components.A);
      B.update(key_data.btn.components.B);
      X.update(key_data.btn.components.X);
      Y.update(key_data.btn.components.Y);
      up.update(key_data.btn.components.up);
      right.update(key_data.btn.components.right);
      down.update(key_data.btn.components.down);
      left.update(key_data.btn.components.left);
    }

    // Override the handle_input function from InputInterface
    // This processes the gamepad input flags and performs actions using the provided parameters
    void handle_input(MotionDataReader& motion_reader, 
                     std::shared_ptr<const MotionSequence>& current_motion, 
                     int& current_frame, 
                     OperatorState& operator_state, 
                     bool& reinitialize_heading, 
                     DataBuffer<HeadingState>& heading_state_buffer, 
                     bool has_planner, 
                     PlannerState& planner_state, 
                     DataBuffer<MovementState>& movement_state_buffer,
                     std::mutex& current_motion_mutex) override {
      

      // Handle safety reset from interface manager
      if (trigger_safety_reset) {
        trigger_safety_reset = false;
        movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
          reinitialize_heading = true;
          auto temp_motion = std::make_shared<MotionSequence>(*current_motion);
          temp_motion->name = "temporary_motion";
          current_motion = temp_motion;
          if (has_planner && planner_state.enabled) {
            planner_state.enabled = false;
            planner_state.initialized = false;
            std::cout << "Safety reset: Planner disabled" << std::endl;
          }
        }
        std::cout << "Safety reset: Returned to reference motion at frame 0" << std::endl;
      }

      // Handle motion control commands
      if (this->motion_prev && !motion_reader.motions.empty()) {
          motion_reader.current_motion_index_ =
              (motion_reader.current_motion_index_ - 1 + motion_reader.motions.size()) % motion_reader.motions.size();
          std::string motion_name;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
            current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
            current_frame = 0;
            motion_name = current_motion->name;
            reinitialize_heading = true;
          }
          std::cout << "Switched to motion " << motion_reader.current_motion_index_ << ": " << motion_name
                    << " (paused at frame 0)" << std::endl;
      }

      if (this->motion_next && !motion_reader.motions.empty()) {
          motion_reader.current_motion_index_ = (motion_reader.current_motion_index_ + 1) % motion_reader.motions.size();
          std::string motion_name;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
            current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
            current_frame = 0;
            motion_name = current_motion->name;
            reinitialize_heading = true;
          }
          std::cout << "Switched to motion " << motion_reader.current_motion_index_ << ": " << motion_name
                    << " (paused at frame 0)" << std::endl;
      }

      if (this->play_motion) {
          if (!operator_state.play) {
              int frame_copy;
              size_t timesteps_copy;
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = true;
                frame_copy = current_frame;
                timesteps_copy = current_motion ? current_motion->timesteps : 0;
              }
              std::cout << "Playing motion " << motion_reader.current_motion_index_ << " from frame " << frame_copy << " to end ("
                        << timesteps_copy << " total frames)" << std::endl;
          }
      }

      if (this->motion_restart) {
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
            current_frame = 0;
            reinitialize_heading = true;
          }
          std::cout << "Reset motion " << motion_reader.current_motion_index_ << " to frame 0 (paused)" << std::endl;
      }

      if (this->stop_control) { operator_state.stop = true; }

      if (this->start_control) { operator_state.start = true; }

      // Handle delta heading controls
      if (this->delta_left) {
          auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
          HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
          double new_delta = current_state.delta_heading + 0.1;
          heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
          std::cout << "Delta heading left: " << new_delta << " rad" << std::endl;
      }

      if (this->delta_right) {
          auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
          HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
          double new_delta = current_state.delta_heading - 0.1;
          heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
          std::cout << "Delta heading right: " << new_delta << " rad" << std::endl;
      }

      // Handle reinitialize command
      if (this->reinitialize) {
        std::lock_guard<std::mutex> lock(current_motion_mutex); 
        reinitialize_heading = true;
      }

      // Handle planner control - copy the toggle state from gamepad
      if (this->use_planner && !has_planner) {
          std::cout << "Planner not loaded - cannot enable" << std::endl;
          this->use_planner = false;
          movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
            current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
            current_frame = 0;
          }
      } else if (has_planner && planner_state.enabled != this->use_planner) {
        planner_state.enabled = this->use_planner;
        if (planner_state.enabled) {
          std::cout << "Planner enabled" << std::endl;
          planner_facing_angle = 0.0;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = false;
          }
          // Wait for planner to be initialized with timeout (5 seconds)
          auto wait_start = std::chrono::steady_clock::now();
          constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
          while (planner_state.enabled) {
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              if (current_motion->name == "planner_motion") {
                std::cout << "[Gamepad] motion name is planner_motion" << std::endl;
                break;
              }
            }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              auto elapsed = std::chrono::steady_clock::now() - wait_start;
              if (elapsed > PLANNER_INIT_TIMEOUT) {
                  std::cerr << "[Gamepad] Planner initialization timeout after 5 seconds" << std::endl;
                  this->use_planner = false;
                  movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
                  {
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    operator_state.play = false;
                    current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
                    current_frame = 0;
                  }
                  break;
              }
              std::cout << "[Gamepad] Waiting for planner to be initialized" << std::endl;
          }
          // Check if planner is enabled and initialized
          if (!planner_state.enabled || !planner_state.initialized) {
              std::cout << "[Gamepad] Planner failed to initialize." << std::endl;
              this->use_planner = false;
              movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
                current_frame = 0;
              }
          } else {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            operator_state.play = true;
          }
        } else {
            std::cout << "Planner disabled" << std::endl;
            planner_state.initialized = false; // Reset planner initialization when disabled
            movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              operator_state.play = false;
              reinitialize_heading = true;
              current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);  // Update current motion directly
              current_frame = 0;
            }
        }
      }

      if (has_planner && planner_state.enabled && planner_state.initialized) {
          
          // Set final movement values based on momentum
          int final_mode = this->planner_use_movement_mode;
          std::array<double, 3> final_movement = {double(cos(planner_moving_direction)), double(sin(planner_moving_direction)), 0.0};
          std::array<double, 3> final_facing_direction = {double(cos(planner_facing_angle)), double(sin(planner_facing_angle)), 0.0};
          double final_speed = this->planner_use_movement_speed;
          double final_height = this->planner_use_height;

          // If left sticks are in the dead zone, idle mode
          if (std::abs(lx) < dead_zone && std::abs(ly) < dead_zone) {
            if constexpr (DEBUG_LOGGING) {
              std::cout << "Both left sticks in the dead zone - Idle mode" << std::endl;
              std::cout << "[GAMEPAD DEBUG] Left stick: lx=" << lx << ", ly=" << ly << std::endl;
            }
            if (is_standing_motion_mode(LocomotionMode(final_mode))) {
              final_mode = static_cast<int>(LocomotionMode::IDLE);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = -1.0f;
              final_height = -1.0f;
            } else {
              final_mode = final_mode;
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = 0;
              final_height = final_height;
            }
          }

          // Handle emergency stop (immediate momentum reset)
          if (this->planner_emergency_stop) {
              planner_use_movement_mode = static_cast<int>(LocomotionMode::IDLE);
              final_mode = static_cast<int>(LocomotionMode::IDLE);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = -1.0f;
              final_height = -1.0f;
              if constexpr (DEBUG_LOGGING) {
                std::cout << "Emergency Stop! Movement momentum reset." << std::endl;
              }
          }

          // Debug: Log final computed values being sent to planner
          if constexpr (DEBUG_LOGGING) {
            static int debug_counter = 0;
            debug_counter++;
            if (debug_counter % 50 == 0) {  // Log every 50 calls to avoid spam
              std::cout << "[GAMEPAD DEBUG] Final planner values:" << std::endl;
              std::cout << "  Mode: " << final_mode << " (0=idle, 1=slow, 2=walk, 3=run, 4=box)" << std::endl;
              std::cout << "  Speed: " << final_speed << std::endl;
              std::cout << "  Height: " << final_height << std::endl;
              std::cout << "  Movement direction: [" << final_movement[0] << ", " << final_movement[1] << ", " << final_movement[2] << "]" << std::endl;
              std::cout << "  Facing direction: [" << final_facing_direction[0] << ", " << final_facing_direction[1] << ", " << final_facing_direction[2] << "]" << std::endl;
            }
          }

          // Update thread-safe buffer (single source of truth for planner thread)
          MovementState mode_state(final_mode, final_movement, final_facing_direction, final_speed, final_height);
          movement_state_buffer.SetData(mode_state);
      }

    }

    // ------------------------------------------------------------------
    // Smoothed analog-stick values (exponential moving average)
    // ------------------------------------------------------------------
    float lx = 0.f;   ///< Left stick horizontal  (smoothed, dead-zone filtered)
    float rx = 0.f;   ///< Right stick horizontal  (smoothed)
    float ry = 0.f;   ///< Right stick vertical    (smoothed)
    float l2 = 0.f;   ///< Left trigger analog     (smoothed)
    float ly = 0.f;   ///< Left stick vertical     (smoothed)

    float smooth = 0.3f;       ///< EMA smoothing factor (0 = no update, 1 = no smoothing).
    float dead_zone = 0.05f;   ///< Analog values below this are zeroed.

    // ------------------------------------------------------------------
    // Edge-detecting button states
    // ------------------------------------------------------------------
    Button R1;      ///< Right bumper
    Button L1;      ///< Left bumper
    Button start;   ///< Start button
    Button select;  ///< Select / back button
    Button R2;      ///< Right trigger (digital)
    Button L2;      ///< Left trigger (digital)
    Button F1;      ///< Function 1 (planner toggle)
    Button F2;      ///< Function 2
    Button A;       ///< A face button (play motion / play in planner)
    Button B;       ///< B face button (restart / emergency stop in planner)
    Button X;       ///< X face button (reinitialize)
    Button Y;       ///< Y face button (reinitialize)
    Button up;      ///< D-pad up
    Button right;   ///< D-pad right (delta heading)
    Button down;    ///< D-pad down
    Button left;    ///< D-pad left  (delta heading)
};
} // namespace unitree::common

#endif