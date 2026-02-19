/**
 * @file gamepad_manager.hpp
 * @brief Planner-centric gamepad manager with ZMQ / ROS2 delegate switching.
 *
 * GamepadManager is a specialised InputInterface that:
 *  - Reads the Unitree wireless gamepad **directly** (no nested Gamepad class).
 *  - Operates **exclusively in planner mode** – there is no reference-motion
 *    cycling.  The planner must always be loaded.
 *  - Can delegate to ZMQEndpointInterface or ROS2InputHandler via D-pad
 *    switching (Up → ZMQ, Down → ROS2, Left/Right → Gamepad).
 *  - Provides keyboard shortcuts (@/# /$ via stdin) for switching and 'O'/'o'
 *    for emergency stop.
 *
 * ## Gamepad Button Mapping (Planner Mode)
 *
 *   Button  | Action
 *   --------|-------
 *   Start   | Start control (enable planner, wait for init, auto-play)
 *   Select  | Emergency stop
 *   X       | Set locomotion to WALK (default speed)
 *   Y       | Set locomotion to RUN (speed 3.0)
 *   B       | Set locomotion to KNEEL_TWO_LEGS → auto-transition to CRAWLING after 2 s
 *   A       | Emergency stop (immediate halt)
 *   L1/R1   | Facing angle ±π/4
 *   L2/R2   | Speed ±0.1 (when speed ≠ −1)
 *   L stick | Movement direction (binned to nearest 30° increment)
 *   R stick | Facing direction (continuous)
 *   D-pad   | Interface switching (Up=ZMQ, Down=ROS2, L/R=Gamepad)
 */

#ifndef GAMEPAD_MANAGER_HPP
#define GAMEPAD_MANAGER_HPP

#include <memory>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>

#include "input_interface.hpp"
#include "zmq_endpoint_interface.hpp"
#include "gamepad.hpp"
#include "../localmotion_kplanner.hpp"  // For LocomotionMode enum

#if HAS_ROS2
#include "ros2_input_handler.hpp"
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class GamepadManager
 * @brief Planner-only gamepad controller with ZMQ / ROS2 delegate switching.
 *
 * When in GAMEPAD mode, button presses and stick positions are translated
 * directly into MovementState commands for the locomotion planner.
 * When in ZMQ or ROS2 mode, all calls are forwarded to the corresponding
 * delegate interface.
 */
class GamepadManager : public InputInterface {
  public:
    // ========================================
    // DEBUG CONTROL FLAG
    // ========================================
    static constexpr bool DEBUG_LOGGING = true;

    enum class ManagedType {
      GAMEPAD = 0,
      ZMQ = 1,
      ROS2 = 2
    };

    GamepadManager(
      const std::string& zmq_host,
      int zmq_port,
      const std::string& zmq_topic,
      bool zmq_conflate,
      bool zmq_verbose
    ) : InputInterface(), zmq_host_(zmq_host), zmq_port_(zmq_port), zmq_topic_(zmq_topic),
        zmq_conflate_(zmq_conflate), zmq_verbose_(zmq_verbose) {
      type_ = InputType::GAMEPAD;  // Default to gamepad mode
      buildInterfaces();
      active_ = ManagedType::GAMEPAD;
      current_ = nullptr;  // Gamepad mode doesn't use delegate
    }

    void update() override {
      // Reset per-frame flags
      emergency_stop_ = false;
      start_control_ = false;
      stop_control_ = false;
      reinitialize_ = false;
      planner_emergency_stop_ = false;

      // Handle stdin shortcuts for switching and emergency stop
      char ch;
      while (ReadStdinChar(ch)) {
        bool is_manager_key = false;
        switch (ch) {
          case '@':
            SetActiveInterface(ManagedType::GAMEPAD);
            is_manager_key = true;
            break;
          case '#':
            SetActiveInterface(ManagedType::ZMQ);
            is_manager_key = true;
            break;
          case '$':
            SetActiveInterface(ManagedType::ROS2);
            is_manager_key = true;
            break;
          case 'o':
          case 'O':
            emergency_stop_ = true;
            is_manager_key = true;
            std::cout << "[GamepadManager] EMERGENCY STOP triggered (O/o key pressed)" << std::endl;
            break;
        }

        if (!is_manager_key && current_) {
          current_->PushStdinChar(ch);
        }
      }

      // Update gamepad data and buttons
      update_gamepad_data(gamepad_data_.RF_RX);

      // D-pad for interface switching
      bool trigger_ZMQ_toggle = false;
      if (up_.on_press) {
        if (active_ != ManagedType::ZMQ) {
            // Switch to ZMQ and auto-enable streaming
            SetActiveInterface(ManagedType::ZMQ);
        } 
        trigger_ZMQ_toggle = true;  
      }
      
      if (down_.on_press) {
        if (active_ != ManagedType::ROS2) {
            // Switch to ROS2
            SetActiveInterface(ManagedType::ROS2);
        }
      }
      
      if (left_.on_press || right_.on_press) {
        if (active_ != ManagedType::GAMEPAD) {
            // Switch to GAMEPAD
            SetActiveInterface(ManagedType::GAMEPAD);
        }
      }

      // Select - Emergency Stop
      if (select_.on_press) { 
        stop_control_ = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Select pressed - Emergency Stop" << std::endl;
        }
      }

      // If not in gamepad mode, also update the active interface
      if (active_ != ManagedType::GAMEPAD && current_) {
        current_->update();
        if (trigger_ZMQ_toggle && zmq_) {
            zmq_->TriggerZMQToggle();
        }
      } else {
        processGamepadPlannerControls();
      }
    }

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
      // Check if planner is loaded (required for GamepadManager)
      if (!has_planner) {
        std::cerr << "[GamepadManager ERROR] Planner not loaded - GamepadManager requires planner. Stopping control." << std::endl;
        operator_state.stop = true;
        return;
      }

      // NOTE: Encoder mode safety check removed - encoder mode is now a property of the motion
      // If a motion has an incompatible encoder mode for gamepad mode, the user should
      // switch to a different motion with a compatible encoder mode

      // Global emergency stop
      if (emergency_stop_) {
        operator_state.stop = true;
      }

      // Handle stop control
      if (stop_control_) { operator_state.stop = true; }

      // If in gamepad mode, handle planner-only controls
      if (active_ == ManagedType::GAMEPAD) {
        handleGamepadPlannerInput(motion_reader, current_motion, current_frame,
                                  operator_state, reinitialize_heading, heading_state_buffer,
                                  planner_state, movement_state_buffer, current_motion_mutex);
      } else {
        // Delegate to ZMQ or ROS2
        if (current_) {
          current_->handle_input(motion_reader, current_motion, current_frame, operator_state,
                                reinitialize_heading, heading_state_buffer, has_planner,
                                planner_state, movement_state_buffer, current_motion_mutex);
        }
      }
    }

    bool HasVR3PointControl() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasVR3PointControl();
      }
      return has_vr_3point_control_;
    }

    bool HasHandJoints() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasHandJoints();
      }
      return has_hand_joints_;
    }

    bool HasExternalTokenState() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->HasExternalTokenState();
      }
      return has_external_token_state_;
    }

    std::pair<bool, std::array<double, 9>> GetVR3PointPosition() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointPosition();
      }
      return InputInterface::GetVR3PointPosition();
    }

    std::pair<bool, std::array<double, 12>> GetVR3PointOrientation() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointOrientation();
      }
      return InputInterface::GetVR3PointOrientation();
    }

    std::array<double, 3> GetVR3PointCompliance() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetVR3PointCompliance();
      }
      return InputInterface::GetVR3PointCompliance();
    }

    std::pair<bool, std::array<double, 7>> GetHandPose(bool is_left) const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetHandPose(is_left);
      }
      return InputInterface::GetHandPose(is_left);
    }

    std::pair<bool, std::vector<double>> GetExternalTokenState() const override {
      if (active_ != ManagedType::GAMEPAD && current_) {
        return current_->GetExternalTokenState();
      }
      return InputInterface::GetExternalTokenState();
    }

    // Receive raw wireless remote data for gamepad
    void UpdateGamepadRemoteData(const uint8_t* buff, size_t size) {
      if (buff == nullptr || size == 0) { return; }
      size_t copy_size = std::min<size_t>(size, sizeof(gamepad_data_.buff));
      std::memcpy(gamepad_data_.buff, buff, copy_size);
    }

    void SetActiveInterface(ManagedType t) {
      for (size_t i = 0; i < order_.size(); ++i) {
        if (order_[i] == t) {
          setActiveIndex(static_cast<int>(i));
          return;
        }
      }
    }

    ManagedType GetActiveInterface() const { return active_; }

  private:
    void buildInterfaces() {
      order_.push_back(ManagedType::GAMEPAD);

      zmq_ = std::make_unique<ZMQEndpointInterface>(
        zmq_host_, zmq_port_, zmq_topic_, zmq_conflate_, zmq_verbose_
      );
      order_.push_back(ManagedType::ZMQ);

#if HAS_ROS2
      ros2_ = std::make_unique<ROS2InputHandler>(true, "g1_deploy_ros2_handler");
      order_.push_back(ManagedType::ROS2);
#endif
    }

    void setActiveIndex(int idx) {
      if (order_.empty()) { return; }
      if (idx < 0) { idx = static_cast<int>(order_.size()) - 1; }
      if (idx >= static_cast<int>(order_.size())) { idx = 0; }

      // Trigger safety reset on all managed interfaces when switching
      TriggerSafetyReset();  // Self (for gamepad mode)
      if (zmq_) zmq_->TriggerSafetyReset();
#if HAS_ROS2
      if (ros2_) ros2_->TriggerSafetyReset();
#endif

      active_index_ = idx;
      active_ = order_[static_cast<size_t>(active_index_)];

      switch (active_) {
        case ManagedType::GAMEPAD:
          current_ = nullptr;  // Gamepad mode is handled directly
          type_ = InputType::GAMEPAD;
          std::cout << "[GamepadManager] Switched to: GAMEPAD (safety reset triggered)" << std::endl;
          break;
        case ManagedType::ZMQ:
          current_ = zmq_.get();
          type_ = InputType::NETWORK;
          std::cout << "[GamepadManager] Switched to: ZMQ (safety reset triggered)" << std::endl;
          break;
        case ManagedType::ROS2:
#if HAS_ROS2
          current_ = ros2_.get();
          type_ = InputType::ROS2;
          std::cout << "[GamepadManager] Switched to: ROS2 (safety reset triggered)" << std::endl;
          break;
#else
          // Should never happen when ROS2 disabled; fall back to gamepad
          current_ = nullptr;
          type_ = InputType::GAMEPAD;
          active_ = ManagedType::GAMEPAD;
          std::cout << "[GamepadManager] ROS2 not available. Falling back to GAMEPAD (safety reset triggered)" << std::endl;
          break;
#endif
      }
    }

    // Update gamepad analog and button data
    void update_gamepad_data(unitree::common::xRockerBtnDataStruct& key_data) {
      lx_ = lx_ * (1 - smooth_) + (std::fabs(key_data.lx) < dead_zone_ ? 0.0f : key_data.lx) * smooth_;
      rx_ = rx_ * (1 - smooth_) + (std::fabs(key_data.rx) < dead_zone_ ? 0.0f : key_data.rx) * smooth_;
      ry_ = ry_ * (1 - smooth_) + (std::fabs(key_data.ry) < dead_zone_ ? 0.0f : key_data.ry) * smooth_;
      l2_ = l2_ * (1 - smooth_) + (std::fabs(key_data.L2) < dead_zone_ ? 0.0f : key_data.L2) * smooth_;
      ly_ = ly_ * (1 - smooth_) + (std::fabs(key_data.ly) < dead_zone_ ? 0.0f : key_data.ly) * smooth_;

      R1_.update(key_data.btn.components.R1);
      L1_.update(key_data.btn.components.L1);
      start_.update(key_data.btn.components.start);
      select_.update(key_data.btn.components.select);
      R2_.update(key_data.btn.components.R2);
      L2_.update(key_data.btn.components.L2);
      F1_.update(key_data.btn.components.F1);
      F2_.update(key_data.btn.components.F2);
      A_.update(key_data.btn.components.A);
      B_.update(key_data.btn.components.B);
      X_.update(key_data.btn.components.X);
      Y_.update(key_data.btn.components.Y);
      up_.update(key_data.btn.components.up);
      right_.update(key_data.btn.components.right);
      down_.update(key_data.btn.components.down);
      left_.update(key_data.btn.components.left);
    }

    // Process gamepad inputs for planner controls (called from update())
    void processGamepadPlannerControls() {
      // Start/Stop buttons
      if (start_.on_press) { 
        start_control_ = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Start pressed" << std::endl;
        }
      }

      // X - Set to Walk mode
      if (X_.on_press) { 
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::WALK); // Walk mode
        planner_use_movement_speed_ = -1; // Default walk speed
        planner_use_height_ = -1.0; // Default walk height
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] X pressed - Walk mode" << std::endl;
        }
      }

      // Y - Set to Run mode
      if (Y_.on_press) { 
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::RUN); // Run mode
        planner_use_movement_speed_ = 3.0; // Default run speed
        planner_use_height_ = -1.0; // Default run height
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Y pressed - Run mode" << std::endl;
        }
      }

      // B - Set to Crawling mode (elbow crawling)
      if (B_.on_press) { 
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS); // kneel two legs mode, will gradually go to crawling mode
        planner_use_movement_speed_ = -1; // Default crawling speed
        planner_use_height_ = 0.4f; // Default crawling height
        kneel_two_legs_start_time_ = std::chrono::steady_clock::now();
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] B pressed - Kneel two legs mode, will gradually go to crawling mode" << std::endl;
        }
      }

      // A - Emergency Stop
      if (A_.on_press) { 
        planner_emergency_stop_ = true; 
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] A pressed - Emergency Stop" << std::endl;
        }
      }

      // L1/R1 - Small angle adjustments (0.02 rad)
      if (L1_.on_press) { 
        planner_facing_angle_ += M_PI/4;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] L1 pressed - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
        }
      }
      if (R1_.on_press) { 
        planner_facing_angle_ -= M_PI/4;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] R1 pressed - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
        }
      }

      // L2/R2 - Change movement speed or height
      if (R2_.on_press) { 
        if (planner_use_movement_speed_ != -1) {
            planner_use_movement_speed_ += 0.1;
        }
      }
      if (L2_.on_press) { 
        if (planner_use_movement_speed_ != -1) {
          planner_use_movement_speed_ -= 0.1;
          }
      }

      // Check if kneel two legs mode has been started for more than 1 second, if so, switch to crawling mode
      if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS) && kneel_two_legs_start_time_.time_since_epoch().count() != 0) {
        auto elapsed = std::chrono::steady_clock::now() - kneel_two_legs_start_time_;
        if (elapsed > std::chrono::seconds(2)) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::CRAWLING); // Crawling mode
          planner_use_movement_speed_ = 0.7; // Default crawling speed
          kneel_two_legs_start_time_ = std::chrono::time_point<std::chrono::steady_clock>{};
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Kneel two legs mode has been started for more than 1 second, switching to crawling mode" << std::endl;
          }
        }
      }


      // Limit movement speed and height based on mode
      switch (planner_use_movement_mode_) {
        case 3:  // Run
          planner_use_movement_speed_ = std::max(planner_use_movement_speed_, 2.5);
          planner_use_movement_speed_ = std::min(planner_use_movement_speed_, 4.5);
          break;
        case 8:  // Crawling
          planner_use_movement_speed_ = std::max(planner_use_movement_speed_, 0.7);
          planner_use_movement_speed_ = std::min(planner_use_movement_speed_, 1.2);
          break;
      }

      // Analog sticks - facing and movement direction
      if (std::abs(rx_) > dead_zone_ || std::abs(ry_) > dead_zone_) {
        planner_facing_angle_ = planner_facing_angle_ - 0.02 * rx_;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Right stick - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
        }
      }

      if (std::abs(lx_) > dead_zone_ || std::abs(ly_) > dead_zone_) {
        // Bin-smooth the angle to nearest π/6 (30 degree) increment
        double raw_angle = atan2(ly_, lx_);
        double bin_size = M_PI / 6.0;
        double binned_angle = std::round(raw_angle / bin_size) * bin_size;
        
        planner_moving_direction_ = binned_angle - M_PI/2 + planner_facing_angle_;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Left stick - Raw angle: " << raw_angle 
                    << " rad, Binned angle: " << binned_angle 
                    << " rad, Moving direction: " << planner_moving_direction_ << " rad" << std::endl;
        }
      }
    }

    // Handle gamepad planner input (called from handle_input())
    void handleGamepadPlannerInput(MotionDataReader& motion_reader,
                                   std::shared_ptr<const MotionSequence>& current_motion,
                                   int& current_frame,
                                   OperatorState& operator_state,
                                   bool& reinitialize_heading,
                                   DataBuffer<HeadingState>& heading_state_buffer,
                                   PlannerState& planner_state,
                                   DataBuffer<MovementState>& movement_state_buffer,
                                   std::mutex& current_motion_mutex) {
      // Handle safety reset from interface manager
      if (CheckAndClearSafetyReset()) {
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
        }
        if (operator_state.start) {
          if (planner_state.enabled && planner_state.initialized) {
            // Planner is already on, keep it as is (don't touch initialized flag)
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              if (current_motion->GetEncodeMode() >= 0) {
                current_motion->SetEncodeMode(0);
              }
              operator_state.play = true;
            }
            auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
            planner_facing_angle_ = std::atan2(current_facing[1], current_facing[0]);
            std::cout << "[GamepadManager] Safety reset: Planner kept enabled with current state" << std::endl;
          } else {
            // Planner was disabled, set initial movement state
            movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), 
                                                        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));

            // Now enable planner
            planner_state.enabled = true;
            planner_facing_angle_ = 0.0;
            std::cout << "[GamepadManager] Planner enabled" << std::endl;

            // Wait for planner to be initialized with timeout (5 seconds)
            auto wait_start = std::chrono::steady_clock::now();
            constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
            while (planner_state.enabled) {
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                if (current_motion->name == "planner_motion") {
                  std::cout << "[GamepadManager] motion name is planner_motion" << std::endl;
                  break;
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              auto elapsed = std::chrono::steady_clock::now() - wait_start;
              if (elapsed > PLANNER_INIT_TIMEOUT) {
                std::cerr << "[GamepadManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
                operator_state.stop = true;
                return;
              }
              std::cout << "[GamepadManager] Waiting for planner to be initialized" << std::endl;
            }

            // Check if planner is enabled and initialized
            if (!planner_state.enabled || !planner_state.initialized) {
              std::cerr << "[GamepadManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
              operator_state.stop = true;
              return;
            }

            // Play motion
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              operator_state.play = true;
            }
          }
        }
        return;
      }

      // Handle control start (Start button)
      if (start_control_) {
        // Start control
        operator_state.start = true;
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
          // Reinitialize base quaternion and reset delta heading
          reinitialize_heading = true;
        }
        
        // Ensure planner is enabled (always required in GamepadManager mode)
        if (!planner_state.enabled) {
          planner_state.enabled = true;
          planner_facing_angle_ = 0.0;
          std::cout << "[GamepadManager] Planner enabled" << std::endl;
        }

        // Wait for planner to be initialized with timeout (5 seconds)
        auto wait_start = std::chrono::steady_clock::now();
        constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
        while (planner_state.enabled) {
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            if (current_motion->name == "planner_motion") {
              std::cout << "[GamepadManager] motion name is planner_motion" << std::endl;
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          auto elapsed = std::chrono::steady_clock::now() - wait_start;
          if (elapsed > PLANNER_INIT_TIMEOUT) {
            std::cerr << "[GamepadManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
            operator_state.stop = true;
            return;
          }
          std::cout << "[GamepadManager] Waiting for planner to be initialized" << std::endl;
        }

        // Check if planner is enabled and initialized
        if (!planner_state.enabled || !planner_state.initialized) {
          std::cerr << "[GamepadManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
          operator_state.stop = true;
          return;
        }

        // Play motion
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = true;
        }
      }

      // Handle reinitialize command
      if (reinitialize_) {
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        reinitialize_heading = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Reinitialized base quaternion and facing angle" << std::endl;
        }
      }

      // If planner is enabled and initialized, send movement commands
      if (planner_state.enabled && planner_state.initialized) {
        int final_mode = planner_use_movement_mode_;
        std::array<double, 3> final_movement = {double(cos(planner_moving_direction_)), 
                                                 double(sin(planner_moving_direction_)), 0.0};
        std::array<double, 3> final_facing_direction = {double(cos(planner_facing_angle_)), 
                                                         double(sin(planner_facing_angle_)), 0.0};
        double final_speed = planner_use_movement_speed_;
        double final_height = planner_use_height_;

        // If left sticks in dead zone, go to idle only for walk/run modes, stay in place for crawl/kneel
        if (std::abs(lx_) < dead_zone_ && std::abs(ly_) < dead_zone_) {
          // If in WALK or RUN mode, go to IDLE
          if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::WALK) || 
              planner_use_movement_mode_ == static_cast<int>(LocomotionMode::RUN)) {
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          }else {
            // Keep current mode, but zero out movement
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = 0.4f;
          }
        }

        // Emergency stop resets to idle
        if (planner_emergency_stop_) {
          if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::WALK) || 
              planner_use_movement_mode_ == static_cast<int>(LocomotionMode::RUN)) {
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          } else {
            // Keep current mode, but zero out movement
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = 0.4f;
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Emergency stop - movement reset" << std::endl;
          }
        }
        
        if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS) ||
        planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL)) {
          final_movement = {0.0f, 0.0f, 0.0f};
          final_speed = 0.0f;
          final_height = 0.4f;
        }

        MovementState mode_state(final_mode, final_movement, final_facing_direction, final_speed, final_height);
        movement_state_buffer.SetData(mode_state);
      }
    }

  private:
    // ------------------------------------------------------------------
    // Owned delegate interfaces (gamepad mode is handled inline, not delegated)
    // ------------------------------------------------------------------
    std::unique_ptr<ZMQEndpointInterface> zmq_;   ///< ZMQ streaming handler.
#if HAS_ROS2
    std::unique_ptr<ROS2InputHandler> ros2_;       ///< ROS 2 teleop handler.
#endif

    InputInterface* current_ = nullptr;  ///< Non-owning pointer to active delegate (nullptr in gamepad mode).

    // ------------------------------------------------------------------
    // Active-selection bookkeeping
    // ------------------------------------------------------------------
    std::vector<ManagedType> order_;         ///< Insertion-order of available modes.
    int active_index_ = 0;                   ///< Index into order_.
    ManagedType active_ = ManagedType::GAMEPAD;  ///< Currently-active mode tag.

    // ZMQ configuration (stored for deferred construction)
    std::string zmq_host_;
    int zmq_port_;
    std::string zmq_topic_;
    bool zmq_conflate_ = false;
    bool zmq_verbose_ = false;

    /// Global emergency-stop flag, set by 'O'/'o' keyboard shortcut.
    bool emergency_stop_ = false;

    // ------------------------------------------------------------------
    // Per-frame gamepad action flags (reset at the start of update())
    // ------------------------------------------------------------------
    bool start_control_ = false;           ///< Start-button pressed this frame.
    bool stop_control_ = false;            ///< Select-button pressed this frame.
    bool reinitialize_ = false;            ///< X/Y-button reinitialize heading.
    bool planner_emergency_stop_ = false;  ///< A-button emergency stop.

    // ------------------------------------------------------------------
    // Gamepad hardware state
    // ------------------------------------------------------------------
    unitree::common::REMOTE_DATA_RX gamepad_data_ = unitree::common::REMOTE_DATA_RX();  ///< Raw 40-byte packet.

    // Smoothed analog stick values
    float lx_ = 0.0f;   ///< Left stick horizontal (smoothed).
    float rx_ = 0.0f;   ///< Right stick horizontal (smoothed).
    float ry_ = 0.0f;   ///< Right stick vertical (smoothed).
    float l2_ = 0.0f;   ///< Left trigger analog (smoothed).
    float ly_ = 0.0f;   ///< Left stick vertical (smoothed).
    float smooth_ = 0.3f;       ///< EMA smoothing factor.
    float dead_zone_ = 0.05f;   ///< Analog dead-zone threshold.

    // Edge-detecting buttons
    unitree::common::Button R1_, L1_, start_, select_, R2_, L2_;
    unitree::common::Button F1_, F2_, A_, B_, X_, Y_;
    unitree::common::Button up_, right_, down_, left_;

    // ------------------------------------------------------------------
    // Planner control state (persists across frames)
    // ------------------------------------------------------------------
    int planner_use_movement_mode_ = static_cast<int>(LocomotionMode::WALK);  ///< Current locomotion mode.
    double planner_use_movement_speed_ = -1.0;   ///< Desired speed (−1 = mode default).
    double planner_use_height_ = -1.0;            ///< Desired body height (−1 = mode default).
    double planner_facing_angle_ = 0.0;           ///< Accumulated facing direction (radians).
    double planner_moving_direction_ = 0.0;       ///< Current movement direction (radians).
    
    /// Timestamp when KNEEL_TWO_LEGS mode was entered; used to auto-transition
    /// to CRAWLING mode after a 2-second delay.
    std::chrono::time_point<std::chrono::steady_clock> kneel_two_legs_start_time_{};
};

#endif // GAMEPAD_MANAGER_HPP


