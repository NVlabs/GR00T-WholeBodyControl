/**
 * @file keyboard_handler.hpp
 * @brief Terminal-based keyboard input handler for the G1 robot.
 *
 * SimpleKeyboard reads single characters from stdin (set to non-blocking,
 * non-canonical mode) and translates them into per-frame action flags.
 *
 * ## Two Operational Modes
 *
 * ### Reference-motion mode (default)
 *   Key | Action
 *   ----|-------
 *   P/p | Previous motion
 *   N/n | Next motion
 *   T/t | Play / resume playback
 *   R/r | Restart motion (frame 0, paused)
 *   ]   | Start control system
 *   O/o | Emergency stop
 *   Q/q | Delta heading left  (−π/12)
 *   E/e | Delta heading right (+π/12)
 *   I/i | Reinitialise heading (recapture IMU)
 *   Enter | Toggle planner mode
 *   Z/z | Toggle encoder mode
 *
 * ### Planner mode (Enter to toggle)
 *   Key | Action
 *   ----|-------
 *   W/S | Move forward / backward
 *   A/D | Adjust-left / adjust-right (slight turn + forward)
 *   ,/. | Strafe left / right
 *   Q/E | Heading left / right (±π/6)
 *   1-8 | Select locomotion mode from current motion set
 *   N/P | Next / previous motion set
 *   9/0 | Decrease / increase speed
 *   -/= | Decrease / increase height
 *   R/` | Emergency stop (reset momentum)
 *   T/t | Play motion
 *   Z/z | Toggle encoder mode
 *
 * Movement uses a momentum system: pressing a direction key sets momentum to
 * 1.0; each frame without input decays it by `momentum_decay_rate`.  Below
 * `momentum_threshold` the robot transitions to IDLE (or stays in the current
 * static pose for squat / boxing sets).
 */

#ifndef KEYBOARD_HANDLER_HPP
#define KEYBOARD_HANDLER_HPP

#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <array>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "input_interface.hpp"

/**
 * @class SimpleKeyboard
 * @brief stdin-based InputInterface for interactive control from a terminal.
 */
class SimpleKeyboard : public InputInterface {
  public:
    // ------------------------------------------------------------------
    // Per-frame action flags  (reset at the start of every update() call)
    // ------------------------------------------------------------------
    bool motion_prev = false;      ///< Switch to previous pre-loaded motion.
    bool motion_next = false;      ///< Switch to next pre-loaded motion.
    bool play_motion = false;      ///< Start / resume motion playback.
    bool motion_restart = false;   ///< Reset current motion to frame 0 (paused).

    bool start_control = false;    ///< Request control-system start.
    bool stop_control = false;     ///< Request emergency stop.

    bool delta_left = false;       ///< Nudge heading left  (−π/12 rad per press).
    bool delta_right = false;      ///< Nudge heading right (+π/12 rad per press).

    bool reinitialize = false;     ///< Recapture the IMU base quaternion.

    // ------------------------------------------------------------------
    // Planner-mode state
    // ------------------------------------------------------------------
    bool use_planner = false;      ///< True while planner mode is active.

    /// Index into the predefined motion-set table
    /// (0 = standing, 1 = squat, 2 = boxing, 3 = styled walk).
    int motion_set_index = 0;
    /// Locomotion modes available in the current motion set.
    std::vector<LocomotionMode> current_motion_set = get_motion_set(motion_set_index);

    // Directional movement flags (per-frame, planner mode only)
    bool planner_move_forward = false;     ///< W key pressed this frame.
    bool planner_move_backward = false;    ///< S key pressed this frame.
    bool planner_move_adj_left = false;    ///< A key – slight left turn + forward.
    bool planner_move_adj_right = false;   ///< D key – slight right turn + forward.
    bool planner_move_left = false;        ///< ',' key – strafe left.
    bool planner_move_right = false;       ///< '.' key – strafe right.
    bool planner_heading_left = false;     ///< E key – heading left (−π/6).
    bool planner_heading_right = false;    ///< Q key – heading right (+π/6).

    LocomotionMode planner_use_movement_mode = LocomotionMode::IDLE;  ///< Current locomotion mode.
    double planner_use_movement_speed = -1;  ///< Desired speed (−1 = mode default).
    double planner_use_height = -1;          ///< Desired body height (−1 = mode default).
    bool planner_emergency_stop = false;     ///< Immediate halt flag (R / ` key).

    bool encoder_mode_toggle = false;  ///< Toggle encoder-mode (Z key) this frame.

    // Persistent planner state
    double planner_facing_angle = 0.0;   ///< Accumulated facing direction (radians).

    /// Movement momentum (0 = stopped, 1 = full speed).
    /// Decays each frame by `momentum_decay_rate` when no movement key is held.
    double movement_momentum = 0.0;
    const double momentum_decay_rate = 0.999;  ///< Per-frame multiplicative decay.
    const double momentum_threshold = 0.1;     ///< Below this, transition to IDLE.

    /**
     * @brief Construct the keyboard handler.
     *
     * Puts stdin into non-canonical, non-echo, non-blocking mode so that
     * individual key-presses can be read without waiting for Enter.
     * The original terminal settings are restored in the destructor.
     */
    explicit SimpleKeyboard() : InputInterface() {
      tcgetattr(STDIN_FILENO, &old_termios_);
      struct termios new_termios = old_termios_;
      new_termios.c_lflag &= ~(ICANON | ECHO);  // Disable line buffering and echo
      tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
      fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);  // Non-blocking reads
      type_ = InputType::KEYBOARD;
    }

    /// Restore original terminal settings on destruction.
    ~SimpleKeyboard() {
      tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
    }

    // Flag to trigger safety reset in handle_input
    bool trigger_safety_reset = false;

    // Override the update function from InputInterface
    void update() override {
      // Check for safety reset trigger from manager
      if (CheckAndClearSafetyReset()) {
        use_planner = false;
        trigger_safety_reset = true;
        std::cout << "[SimpleKeyboard] Safety reset triggered: will disable planner and return to reference motion" << std::endl;
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
      planner_move_forward = false;
      planner_move_backward = false;
      planner_move_adj_left = false;
      planner_move_adj_right = false;
      planner_move_left = false;
      planner_move_right = false;
      planner_heading_left = false;
      planner_heading_right = false;
      planner_emergency_stop = false;
      encoder_mode_toggle = false;

      // Read keyboard input (using shared buffered reading)
      char ch;
      while (ReadStdinChar(ch)) {
        if (use_planner) {
            switch (ch) {
                case 'r':
                case 'R':
                case '`':
                case '~': planner_emergency_stop = true; break; // Emergency stop - immediate halt
                case '1':
                case '!': if (current_motion_set.size() > 0) planner_use_movement_mode = current_motion_set[0]; break; // Use standing motion set
                case '2':
                case '@': if (current_motion_set.size() > 1) planner_use_movement_mode = current_motion_set[1]; break; // Use squat motion set
                case '3':
                case '#': if (current_motion_set.size() > 2) planner_use_movement_mode = current_motion_set[2]; break; // Use boxing motion set
                case '4':
                case '$': if (current_motion_set.size() > 3) planner_use_movement_mode = current_motion_set[3]; break; // Use boxing or idele squat mode
                case '5':
                case '%': if (current_motion_set.size() > 4) planner_use_movement_mode = current_motion_set[4]; break; // Use kneel two legs mode
                case '6':
                case '^': if (current_motion_set.size() > 5) planner_use_movement_mode = current_motion_set[5]; break; // Use kneel mode
                case '7':
                case '&': if (current_motion_set.size() > 6) planner_use_movement_mode = current_motion_set[6]; break; // Use lying face down mode
                case '8':
                case '*': if (current_motion_set.size() > 7) planner_use_movement_mode = current_motion_set[7]; break; // Use crawling mode
                case '-':
                case '_': if(planner_use_height != -1) { planner_use_height = planner_use_height - 0.1; } break; // Delta height down
                case '=':
                case '+': if(planner_use_height != -1) { planner_use_height = planner_use_height + 0.1; } break; // Delta height up
                case '9':
                case '(': if(planner_use_movement_speed != -1) { planner_use_movement_speed = planner_use_movement_speed - 0.1; } break; // Delta speed up
                case '0':
                case ')': if(planner_use_movement_speed != -1) { planner_use_movement_speed = planner_use_movement_speed + 0.1; } break; // Delta speed down
                case 'n':
                case 'N': motion_set_index = (motion_set_index + 1) % 4; 
                          current_motion_set = get_motion_set(motion_set_index); 
                          planner_use_movement_mode = current_motion_set[0]; 
                          std::cout << "Motion set: " << motion_set_index << std::endl;
                          if (motion_set_index == 1) {
                            planner_use_height = 0.8;
                          }
                          break; // Next motion set
                case 'p':
                case 'P': motion_set_index = (motion_set_index - 1 + 4) % 4; 
                          current_motion_set = get_motion_set(motion_set_index); 
                          planner_use_movement_mode = current_motion_set[0]; 
                          std::cout << "Motion set: " << motion_set_index << std::endl;
                          if (motion_set_index == 1) {
                            planner_use_height = 0.8;
                          }
                          break; // Previous motion set
                case 'j':
                case 'J': delta_left = true; break; // Delta heading left
                case 'l':
                case 'L': delta_right = true; break; // Delta heading right
                case 'w':
                case 'W': planner_move_forward = true; break; // Move forward
                case 's':
                case 'S': planner_move_backward = true; break; // Move backward
                case ',':
                case '<': planner_move_left = true; break; // Move left
                case '.': 
                case '>': planner_move_right = true; break; // Move right
                case 'a':
                case 'A': planner_move_adj_left = true; break; // Move adj left
                case 'd':
                case 'D': planner_move_adj_right = true; break; // Move adj right
                case 'e':
                case 'E': planner_heading_left = true; break; // Delta heading left (-0.1)
                case 'q':
                case 'Q': planner_heading_right = true; break; // Delta heading right (+0.1)
                case ']': start_control = true; break; // Start control system
                case 'o':
                case 'O': stop_control = true; break; // Stop/Exit
                case '\n': use_planner = !use_planner; break; // Use planner
                case 'i':
                case 'I': reinitialize = true; break; // Reinitialize base quaternion and delta heading
                case 't':
                case 'T': play_motion = true; break; // Play motion to end
                case 'z':
                case 'Z': encoder_mode_toggle = true; break; // Toggle encoder mode
            }

            // Limit movement speed and height to the range of the movement mode
            if (is_standing_motion_mode(planner_use_movement_mode)) {
              planner_use_height = -1.0;
            } else {
              planner_use_height = std::max(planner_use_height, 0.2);
              planner_use_height = std::min(planner_use_height, 0.8);
            }

            if (is_static_motion_mode(planner_use_movement_mode)) {
              planner_use_movement_speed = -1.0;
            } else {
              if (planner_use_movement_mode == LocomotionMode::SLOW_WALK) {
                planner_use_movement_speed = std::max(planner_use_movement_speed, 0.2);
                planner_use_movement_speed = std::min(planner_use_movement_speed, 0.8);
              } else if (planner_use_movement_mode == LocomotionMode::RUN) {
                planner_use_movement_speed = std::max(planner_use_movement_speed, 1.5);
                planner_use_movement_speed = std::min(planner_use_movement_speed, 3.0);
              } else if (planner_use_movement_mode == LocomotionMode::CRAWLING) {
                planner_use_movement_speed = std::max(planner_use_movement_speed, 0.4);
                planner_use_movement_speed = std::min(planner_use_movement_speed, 1.5);
              } else if (planner_use_movement_mode == LocomotionMode::ELBOW_CRAWLING) {
                planner_use_movement_speed = std::max(planner_use_movement_speed, 0.7);
                planner_use_movement_speed = std::min(planner_use_movement_speed, 1.5);
              } else if (planner_use_movement_mode == LocomotionMode::WALK_BOXING || 
                          planner_use_movement_mode == LocomotionMode::LEFT_PUNCH || 
                          planner_use_movement_mode == LocomotionMode::RIGHT_PUNCH || 
                          planner_use_movement_mode == LocomotionMode::RANDOM_PUNCH ||
                          planner_use_movement_mode == LocomotionMode::LEFT_HOOK ||
                          planner_use_movement_mode == LocomotionMode::RIGHT_HOOK) {
                planner_use_movement_speed = std::max(planner_use_movement_speed, 0.7);
                planner_use_movement_speed = std::min(planner_use_movement_speed, 1.5);
              }
              else {
                planner_use_movement_speed = -1;
              }
            }
        } else {
          switch (ch) {
            case 'p':
            case 'P': motion_prev = true; break; // Previous motion
            case 'n':
            case 'N': motion_next = true; break; // Next motion
            case 't':
            case 'T': play_motion = true; break; // Play motion to end
            case 'r':
            case 'R': motion_restart = true; break; // Restart motion
            case ']': start_control = true; break; // Start control system
            case 'o':
            case 'O': stop_control = true; break; // Stop/Exit
            case 'q':
            case 'Q': delta_left = true; break; // Delta heading left (-0.1)
            case 'e':
            case 'E': delta_right = true; break; // Delta heading right (+0.1)
            case 'i':
            case 'I': reinitialize = true; break; // Reinitialize base quaternion and delta heading
            case '\n': use_planner = !use_planner; break; // Use planner
            case 'z':
            case 'Z': encoder_mode_toggle = true; break; // Toggle encoder mode
          }
        }
        
      }
    }

    // Override the handle_input function from InputInterface
    // This processes the keyboard input flags and performs actions using the provided parameters
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

      // Handle encoder mode toggle
      if (encoder_mode_toggle) {
        int current_encoder_mode = current_motion->GetEncodeMode();
        if (current_encoder_mode == -2) {
          std::cout << "⚠ No encoder configured - cannot toggle encoder mode" << std::endl;
        } else if (current_encoder_mode == -1) {
          std::cout << "⚠ No encoder loaded - cannot toggle encoder mode" << std::endl;
        } else {
          int new_encoder_mode = (current_encoder_mode == 0) ? 1 : 0;  // Toggle between mode 0 and 1
          std::cout << "Encoder mode: " << (new_encoder_mode == 0 ? "mode 0" : "mode 1") << std::endl;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            current_motion->SetEncodeMode(new_encoder_mode);
          }
        }
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
          double new_delta = current_state.delta_heading + M_PI / 12;
          heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
          std::cout << "Delta heading left: " << new_delta << " rad" << std::endl;
      }

      if (this->delta_right) {
          auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
          HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
          double new_delta = current_state.delta_heading - M_PI / 12;
          heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
          std::cout << "Delta heading right: " << new_delta << " rad" << std::endl;
      }

      // Handle reinitialize command
      if (this->reinitialize) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        reinitialize_heading = true;
      }

      // Handle planner control - copy the toggle state from keyboard
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
                std::cout << "[Keyboard] motion name is planner_motion" << std::endl;
                break;
              }
            }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              auto elapsed = std::chrono::steady_clock::now() - wait_start;
              if (elapsed > PLANNER_INIT_TIMEOUT) {
                  std::cerr << "[Keyboard] Planner initialization timeout after 5 seconds" << std::endl;
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
              std::cout << "[Keyboard] Waiting for planner to be initialized" << std::endl;
          }
          // Check if planner is enabled and initialized
          if (!planner_state.enabled || !planner_state.initialized) {
              std::cout << "[Keyboard] Planner failed to initialize." << std::endl;
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
            movement_momentum = 0.0;
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
          // Handle emergency stop (immediate momentum reset)
          if (this->planner_emergency_stop) {
              movement_momentum = 0.0;
              std::cout << "Emergency Stop! Movement momentum reset." << std::endl;
          }
          
          // Get current state from buffer (single source of truth)
          auto current_mode_data = movement_state_buffer.GetDataWithTime();
          int current_mode = current_mode_data.data ? current_mode_data.data->locomotion_mode : static_cast<int>(LocomotionMode::IDLE);
          std::array<double, 3> current_movement = current_mode_data.data ? current_mode_data.data->movement_direction : std::array<double, 3>{0.0, 0.0, 0.0};
          std::array<double, 3> current_facing = current_mode_data.data ? current_mode_data.data->facing_direction : std::array<double, 3>{1.0, 0.0, 0.0};
          
          // Use local variables for all processing - avoid updating globals until the end
          std::array<double, 3> local_target_movement = current_movement;
          std::array<double, 3> local_facing_direction = current_facing;
          
          // Handle heading controls (independent of movement momentum)
          if (this->planner_heading_left) {
              planner_facing_angle -= M_PI / 6;
              local_facing_direction[0] = cos(planner_facing_angle);
              local_facing_direction[1] = sin(planner_facing_angle);
          }
          if (this->planner_heading_right) {
              planner_facing_angle += M_PI / 6;
              local_facing_direction[0] = cos(planner_facing_angle);
              local_facing_direction[1] = sin(planner_facing_angle);
          }
          bool movement_key_pressed = false;
          
          if (this->planner_move_forward && !is_static_motion_mode(planner_use_movement_mode)) {
              // Just indicate we want to move - actual mode will be determined later based on momentum
              local_target_movement[0] = local_facing_direction[0];
              local_target_movement[1] = local_facing_direction[1];
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          if (this->planner_move_backward && !is_static_motion_mode(planner_use_movement_mode)) {
              local_target_movement[0] = -local_facing_direction[0];
              local_target_movement[1] = -local_facing_direction[1];
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          if (this->planner_move_adj_left && !is_static_motion_mode(planner_use_movement_mode)) {
              planner_facing_angle += 0.1;
              local_facing_direction[0] = cos(planner_facing_angle);
              local_facing_direction[1] = sin(planner_facing_angle);
              local_facing_direction[2] = 0.0f;
              local_target_movement[0] = local_facing_direction[0];
              local_target_movement[1] = local_facing_direction[1];
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          if (this->planner_move_adj_right && !is_static_motion_mode(planner_use_movement_mode)) {
              planner_facing_angle -= 0.1;
              local_facing_direction[0] = cos(planner_facing_angle);
              local_facing_direction[1] = sin(planner_facing_angle);
              local_facing_direction[2] = 0.0f;
              local_target_movement[0] = local_facing_direction[0];
              local_target_movement[1] = local_facing_direction[1];
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          if (this->planner_move_left && !is_static_motion_mode(planner_use_movement_mode)) {
              local_target_movement[0] = -sin(planner_facing_angle);
              local_target_movement[1] = cos(planner_facing_angle);
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          if (this->planner_move_right && !is_static_motion_mode(planner_use_movement_mode)) {
              local_target_movement[0] = sin(planner_facing_angle);
              local_target_movement[1] = -cos(planner_facing_angle);
              local_target_movement[2] = 0.0f;
              movement_momentum = 1.0;  // Full momentum
              movement_key_pressed = true;
          }
          
          // Handle mode changes - just update the mode, keep momentum
          // No need to reset momentum when switching between walk/run modes
          
          // Apply momentum decay if no movement key is pressed
          if (!movement_key_pressed) {
              movement_momentum *= momentum_decay_rate;
          }
          
          // Set final movement values based on momentum
          int final_mode;
          std::array<double, 3> final_movement;
          double final_speed;
          double final_height;
          if (movement_momentum > momentum_threshold) {
              // Determine final mode based on keyboard setting only when we have momentum
              final_mode = static_cast<int>(this->planner_use_movement_mode);
              final_movement = local_target_movement;
              final_speed = this->planner_use_movement_speed;
              final_height = this->planner_use_height;
          } else {
            if (motion_set_index == 1) {
              // Below threshold - maintain current squat mode
              final_mode = static_cast<int>(this->planner_use_movement_mode);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = -1.0f;
              final_height = this->planner_use_height;
              if (this->planner_use_movement_mode == LocomotionMode::CRAWLING) {
                final_speed = 0;
              } else if (this->planner_use_movement_mode == LocomotionMode::ELBOW_CRAWLING) {
                final_speed = 0;
              }
            } else if (motion_set_index == 2) {
              // Below threshold - maintain current boxing mode
              final_mode = static_cast<int>(this->planner_use_movement_mode);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = -1.0f;
              final_height = -1.0f;
              if (this->planner_use_movement_mode == LocomotionMode::WALK_BOXING) {
                final_speed = 0;
              } else if (this->planner_use_movement_mode == LocomotionMode::LEFT_PUNCH) {
                final_speed = 0;
                final_movement = local_facing_direction;
              } else if (this->planner_use_movement_mode == LocomotionMode::RIGHT_PUNCH) {
                final_speed = 0;
                final_movement = local_facing_direction;
              } else if (this->planner_use_movement_mode == LocomotionMode::RANDOM_PUNCH) {
                final_speed = 0;
              } else if (this->planner_use_movement_mode == LocomotionMode::LEFT_HOOK) {
                final_speed = 0;
                final_movement = local_facing_direction;
              } else if (this->planner_use_movement_mode == LocomotionMode::RIGHT_HOOK) {
                final_speed = 0;
                final_movement = local_facing_direction;
              }
            } else if (motion_set_index == 3) {
              // Below threshold - maintain current styled walking mode
              final_mode = static_cast<int>(this->planner_use_movement_mode);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = 0.0f;
              final_height = -1.0f;
            } else {
              // Below threshold - switch to IDLE
              final_mode = static_cast<int>(LocomotionMode::IDLE);
              final_movement = {0.0f, 0.0f, 0.0f};
              final_speed = -1.0f;
              final_height = -1.0f;
            }
          }
         
          // Update thread-safe buffer (single source of truth for planner thread)
          MovementState mode_state(final_mode, final_movement, local_facing_direction, final_speed, final_height);
          movement_state_buffer.SetData(mode_state);
      }
    }


  private:
    struct termios old_termios_;
};

#endif // KEYBOARD_HANDLER_HPP
