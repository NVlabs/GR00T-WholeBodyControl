/**
 * @file zmq_manager.hpp
 * @brief Network-only input manager that switches between planner mode and
 *        streamed-motion mode, both driven by ZMQ topics.
 *
 * ZMQManager subscribes to **three** ZMQ topics on the same host:port:
 *
 *   Topic      | Purpose
 *   -----------|--------
 *   command    | High-level control (start / stop / mode switch).
 *              | Wire format: `{ start: bool, stop: bool, planner: bool, delta_heading?: f32 }`
 *   planner    | Per-frame locomotion commands (mode, movement, facing, speed, height,
 *              | optional upper-body / hand / VR data).  Active in PLANNER mode.
 *   pose       | Streamed motion frames (joint_pos, joint_vel, body_quat, …).
 *              | Active in STREAMED_MOTION mode, handled by an internal ZMQEndpointInterface.
 *
 * ## Mode Switching
 *
 * The `planner` field in the command message selects the mode:
 *   - `planner = true`  → PLANNER mode (movement commands from the planner topic).
 *   - `planner = false` → STREAMED_MOTION mode (pose data from the pose topic).
 *
 * On each mode switch, safety resets are triggered and the planner buffer is
 * cleared to prevent stale commands from leaking across modes.
 *
 * ## Planner Timeout
 *
 * If no planner message arrives within 1 second (PLANNER_TIMEOUT), the manager
 * automatically resets the locomotion to IDLE and clears upper-body / hand-joint
 * control flags.
 *
 * ## Keyboard Shortcuts (via stdin)
 *
 *   Key  | Action
 *   -----|-------
 *   O/o  | Emergency stop
 *   g/G, h/H | Left-hand compliance ±0.1
 *   b/B, v/V | Right-hand compliance ±0.1
 *   x/X, c/C | Hand max-close ratio ±0.1
 */

#ifndef ZMQ_MANAGER_HPP
#define ZMQ_MANAGER_HPP

#include <memory>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>
#include <mutex>

#include "input_interface.hpp"
#include "input_command.hpp"
#include "zmq_endpoint_interface.hpp"
#include "zmq_packed_message_subscriber.hpp"
#include "../localmotion_kplanner.hpp"  // For LocomotionMode enum
#include "../math_utils.hpp"  // For normalize_vector

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class ZMQManager
 * @brief InputInterface that manages two ZMQ-driven modes:
 *        PLANNER (locomotion commands) and STREAMED_MOTION (pose data).
 *
 * Internally owns a ZMQEndpointInterface for streamed-motion mode and two
 * ZMQPackedMessageSubscriber instances for the command and planner topics.
 */
class ZMQManager : public InputInterface {
  public:
    static constexpr bool DEBUG_LOGGING = false;

    enum class ManagedMode {
      PLANNER = 0,         // Planner-only mode (self-managed planner topic)
      STREAMED_MOTION = 1  // ZMQ streamed motion mode (pose topic via ZMQEndpointInterface)
    };

    ZMQManager(
      const std::string& zmq_host,
      int zmq_port,
      const std::string& pose_topic = "pose",
      const std::string& command_topic = "command",
      const std::string& planner_topic = "planner",
      bool zmq_conflate = false,
      bool zmq_verbose = false
    ) : InputInterface(), 
        zmq_host_(zmq_host), 
        zmq_port_(zmq_port), 
        pose_topic_(pose_topic),
        command_topic_(command_topic),
        planner_topic_(planner_topic),
        zmq_conflate_(zmq_conflate), 
        zmq_verbose_(zmq_verbose) {
      
      type_ = InputType::NETWORK;
      active_mode_ = ManagedMode::PLANNER;  // Default to planner mode
      
      // Create pose interface (for streamed motion mode)
      pose_interface_ = std::make_unique<ZMQEndpointInterface>(
        zmq_host_, zmq_port_, pose_topic_, zmq_conflate_, zmq_verbose_
      );
      
      // Create command subscriber
      command_subscriber_ = std::make_unique<ZMQPackedMessageSubscriber>(
        zmq_host_, zmq_port_, command_topic_,
        /*timeout_ms=*/100,
        zmq_verbose_,
        /*use_conflate=*/false,
        /*rcv_hwm=*/3
      );
      
      command_subscriber_->SetOnDecodedMessage(
        [this](const std::string& topic,
               const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
               const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
          this->OnCommandReceived(topic, hdr, bufs);
        }
      );
      
      command_subscriber_->Start();
      
      // Create planner subscriber
      planner_subscriber_ = std::make_unique<ZMQPackedMessageSubscriber>(
        zmq_host_, zmq_port_, planner_topic_,
        /*timeout_ms=*/100,
        zmq_verbose_,
        /*use_conflate=*/false, 
        /*rcv_hwm=*/3
      );
      
      planner_subscriber_->SetOnDecodedMessage(
        [this](const std::string& topic,
               const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
               const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
          this->OnPlannerReceived(topic, hdr, bufs);
        }
      );
      
      planner_subscriber_->Start();
      
      std::cout << "[ZMQManager] Initialized (default: PLANNER mode)" << std::endl;
      std::cout << "  - Host: " << zmq_host_ << ":" << zmq_port_ << std::endl;
      std::cout << "  - Command topic: '" << command_topic_ << "' (start/stop/mode)" << std::endl;
      std::cout << "    Format: { start: bool, stop: bool, planner: bool }" << std::endl;
      std::cout << "  - Planner topic: '" << planner_topic_ << "' (movement)" << std::endl;
      std::cout << "  - Pose topic: '" << pose_topic_ << "' (streamed motion)" << std::endl;
    }
    
    ~ZMQManager() {
      if (command_subscriber_) command_subscriber_->Stop();
      if (planner_subscriber_) planner_subscriber_->Stop();
    }

    void update() override {
      // Reset per-frame flags
      emergency_stop_ = false;
      start_control_ = false;
      stop_control_ = false;
      
      // Handle stdin shortcuts
      char ch;
      while (ReadStdinChar(ch)) {
        bool is_manager_key = false;
        switch (ch) {
          case 'o':
          case 'O':
            emergency_stop_ = true;
            is_manager_key = true;
            std::cout << "[ZMQManager] EMERGENCY STOP (O/o)" << std::endl;
            break;
          // Global compliance controls - work across ALL modes
          case 'g':
          case 'G':
            // Increase left hand compliance by 0.1
            AdjustLeftHandCompliance(0.1);
            is_manager_key = true;
            break;
          case 'h':
          case 'H':
            // Decrease left hand compliance by 0.1
            AdjustLeftHandCompliance(-0.1);
            is_manager_key = true;
            break;
          case 'b':
          case 'B':
            // Increase right hand compliance by 0.1
            AdjustRightHandCompliance(0.1);
            is_manager_key = true;
            break;
          case 'v':
          case 'V':
            // Decrease right hand compliance by 0.1
            AdjustRightHandCompliance(-0.1);
            is_manager_key = true;
            break;
          // Global hand max close ratio controls (x/c keys)
          case 'x':
          case 'X':
            // Increase max close ratio by 0.1 (allow hands to close more)
            AdjustMaxCloseRatio(0.1);
            is_manager_key = true;
            break;
          case 'c':
          case 'C':
            // Decrease max close ratio by 0.1 (keep hands more open)
            AdjustMaxCloseRatio(-0.1);
            is_manager_key = true;
            break;
        }

        // Pass other keys to pose interface (only in streamed motion mode)
        if (!is_manager_key && active_mode_ == ManagedMode::STREAMED_MOTION && pose_interface_) {
          pose_interface_->PushStdinChar(ch);
        }
      }

      // Translate received command to control flags and handle mode switching
      bool trigger_zmq_toggle = false;
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (latest_command_.valid) {
          // Set control flags (already accumulated in callback)
          if (latest_command_.start) {
            start_control_ = true;
          }
          if (latest_command_.stop) {
            stop_control_ = true;
          }

          // Handle mode switching
          ManagedMode new_mode = latest_command_.planner ? ManagedMode::PLANNER : ManagedMode::STREAMED_MOTION;
          
          if (new_mode != active_mode_) {
            // Trigger safety reset on mode switch
            TriggerSafetyReset();
            if (pose_interface_) {
              pose_interface_->TriggerSafetyReset();
            }

            if (new_mode == ManagedMode::PLANNER) {
              std::cout << "[ZMQManager] Switched to: PLANNER mode (safety reset)" << std::endl;
              if (latest_planner_message_.valid) {
                constexpr auto PLANNER_MESSAGE_TIMEOUT = std::chrono::milliseconds(100);
                auto time_since_last_planner = std::chrono::steady_clock::now() - latest_planner_message_.timestamp;
                if (time_since_last_planner < PLANNER_MESSAGE_TIMEOUT) {
                  // Valid planner message within timeout - use it
                  // Update upper body control state based on this message
                  has_upper_body_control_ = latest_planner_message_.upper_body_position.has_value();

                  // Update hand joints control state based on this message
                  has_hand_joints_ = latest_planner_message_.left_hand_joints.has_value() || 
                                     latest_planner_message_.right_hand_joints.has_value();
                }
              }
            } else if (new_mode == ManagedMode::STREAMED_MOTION) {
              std::cout << "[ZMQManager] Switched to: STREAMED MOTION mode (safety reset)" << std::endl;
              trigger_zmq_toggle = true;

              // Clear planner buffer when switching away from planner mode
              {
                std::lock_guard<std::mutex> lock(planner_mutex_);
                latest_planner_message_.valid = false;
                latest_planner_message_.timestamp = {};
                is_planner_ready_ = false;
                switch_from_teleop_to_planner_ = true;
              }
              std::cout << "[ZMQManager] Cleared planner buffer" << std::endl;
            }
          }

          // Clear valid flag - next callback will start fresh accumulation
          active_mode_ = new_mode;
          latest_command_.valid = false;
        }
      }

      // Update active interface based on mode
      if (active_mode_ == ManagedMode::STREAMED_MOTION && pose_interface_) {
        // In streamed motion mode: update pose interface
        pose_interface_->update();
        if (trigger_zmq_toggle) {
          pose_interface_->TriggerZMQToggle();
          std::cout << "[ZMQManager] ZMQ streaming enabled" << std::endl;
        }
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
      if (!has_planner) {
        std::cerr << "[ZMQCommandManager ERROR] Planner not available in planner mode" << std::endl;
        operator_state.stop = true;
        return;
      }
      // Emergency stop
      if (emergency_stop_) {
        operator_state.stop = true;
        if (planner_state.enabled) {
          planner_state.enabled = false;
          planner_state.initialized = false;
        }
        
        // Clear planner buffer on emergency stop
        {
          std::lock_guard<std::mutex> lock(planner_mutex_);
          latest_planner_message_.valid = false;
          latest_planner_message_.timestamp = {};
        }
        // Clear upper body control state
        has_upper_body_control_ = false;
        
        // Clear hand joints control state
        has_hand_joints_ = false;
        
        return;
      }

      // Handle stop control
      if (stop_control_) {
        operator_state.stop = true;
        if (planner_state.enabled) {
          planner_state.enabled = false;
          planner_state.initialized = false;
        }
        
        // Clear planner buffer on stop
        {
          std::lock_guard<std::mutex> lock(planner_mutex_);
          latest_planner_message_.valid = false;
          latest_planner_message_.timestamp = {};
        }
        // Clear upper body control state
        has_upper_body_control_ = false;
        
        // Clear hand joints control state
        has_hand_joints_ = false;
      }

      // Delegate based on current mode
      if (active_mode_ == ManagedMode::PLANNER) {
        // Planner mode: handle planner input ourselves
        handlePlannerInput(motion_reader, current_motion, current_frame,
                          operator_state, reinitialize_heading,
                          heading_state_buffer,
                          has_planner, planner_state, movement_state_buffer,
                          current_motion_mutex);
      } else {
        // Streamed motion mode: delegate to pose interface
        if (pose_interface_) {
          pose_interface_->handle_input(motion_reader, current_motion, current_frame,
                                       operator_state, reinitialize_heading,
                                       heading_state_buffer,
                                       has_planner, planner_state, movement_state_buffer,
                                       current_motion_mutex);
        }
      }
    }

    // Forward getters to pose interface when in streamed motion mode
    bool HasVR3PointControl() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->HasVR3PointControl();
      }
      return has_vr_3point_control_;
    }

    bool HasHandJoints() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->HasHandJoints();
      }
      return has_hand_joints_;
    }

    bool HasExternalTokenState() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->HasExternalTokenState();
      }
      return has_external_token_state_;
    }

    std::pair<bool, std::array<double, 9>> GetVR3PointPosition() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->GetVR3PointPosition();
      }
      return InputInterface::GetVR3PointPosition();
    }

    std::pair<bool, std::array<double, 12>> GetVR3PointOrientation() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->GetVR3PointOrientation();
      }
      return InputInterface::GetVR3PointOrientation();
    }

    std::array<double, 3> GetVR3PointCompliance() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->GetVR3PointCompliance();
      }
      return InputInterface::GetVR3PointCompliance();
    }

    std::pair<bool, std::array<double, 7>> GetHandPose(bool is_left) const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->GetHandPose(is_left);
      }
      return InputInterface::GetHandPose(is_left);
    }

    std::pair<bool, std::vector<double>> GetExternalTokenState() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION || (!is_planner_ready_ && switch_from_teleop_to_planner_)) && pose_interface_) {
        return pose_interface_->GetExternalTokenState();
      }
      return InputInterface::GetExternalTokenState();
    }

    std::optional<std::chrono::steady_clock::time_point> GetLastUpdateTime() const override {
      if ((active_mode_ == ManagedMode::STREAMED_MOTION) && pose_interface_) {
        return pose_interface_->GetLastUpdateTime();
      }
      return InputInterface::GetLastUpdateTime();
    }

  private:
    // Handle planner mode input (similar to GamepadManager::handleGamepadPlannerInput)
    void handlePlannerInput(MotionDataReader& motion_reader,
                           std::shared_ptr<const MotionSequence>& current_motion,
                           int& current_frame,
                           OperatorState& operator_state,
                           bool& reinitialize_heading,
                           DataBuffer<HeadingState>& heading_state_buffer,
                           bool has_planner,
                           PlannerState& planner_state,
                           DataBuffer<MovementState>& movement_state_buffer,
                           std::mutex& current_motion_mutex) {
      
      // Handle safety reset from interface manager (same as GamepadManager)
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
              if (current_motion->GetEncodeMode() == 1) {
                current_motion->SetEncodeMode(0);
              }
              operator_state.play = true;
            }
            auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
            std::cout << "[ZMQManager] Safety reset: Planner kept enabled with current state" << std::endl;
          } else {
            // Planner was disabled, set initial movement state
            movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), 
                                                        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));

            // Now enable planner
            planner_state.enabled = true;
            std::cout << "[ZMQManager] Planner enabled" << std::endl;

            // Wait for planner to be initialized with timeout (5 seconds)
            auto wait_start = std::chrono::steady_clock::now();
            constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
            while (planner_state.enabled) {
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                if (current_motion->name == "planner_motion") {
                  break;
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              auto elapsed = std::chrono::steady_clock::now() - wait_start;
              if (elapsed > PLANNER_INIT_TIMEOUT) {
                std::cerr << "[ZMQCommandManager ERROR] Planner initialization timeout after 5 seconds" << std::endl;
                operator_state.stop = true;
                return;
              }
              std::cout << "[ZMQManager] Waiting for planner to be initialized" << std::endl;
            }

            // Check if planner is enabled and initialized
            if (!planner_state.enabled || !planner_state.initialized) {
              std::cerr << "[ZMQCommandManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
              operator_state.stop = true;
              return;
            }

            is_planner_ready_ = true;

            // Play motion
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex);
              operator_state.play = true;
            }
          }
        }
        return;
      }

      // Handle start control
      if (start_control_ && !operator_state.start) {
        operator_state.start = true;
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
          reinitialize_heading = true;
        }

        // Ensure planner is enabled
        if (!planner_state.enabled) {
          planner_state.enabled = true;
          std::cout << "[ZMQManager] Planner enabled" << std::endl;
        }
        
        // Wait for initialization
        auto wait_start = std::chrono::steady_clock::now();
        constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
        while (planner_state.enabled) {
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            if (current_motion->name == "planner_motion") {
              std::cout << "[ZMQManager] motion name is planner_motion" << std::endl;
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          auto elapsed = std::chrono::steady_clock::now() - wait_start;
          if (elapsed > PLANNER_INIT_TIMEOUT) {
            std::cerr << "[ZMQCommandManager ERROR] Planner initialization timeout" << std::endl;
            operator_state.stop = true;
            return;
          }
          std::cout << "[ZMQManager] Waiting for planner to be initialized" << std::endl;
        }
        
        // Check if planner is enabled and initialized
        if (!planner_state.enabled || !planner_state.initialized) {
          std::cerr << "[ZMQCommandManager ERROR] Planner failed to initialize. Stopping control." << std::endl;
          operator_state.stop = true;
          return;
        }
        
        is_planner_ready_ = true;

        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = true;
        }
      }

      // Apply planner commands if planner is ready
      if (planner_state.enabled && planner_state.initialized) {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        
        // Check for planner timeout (1 second)
        constexpr auto PLANNER_TIMEOUT = std::chrono::milliseconds(1000);
        auto time_since_last_planner = std::chrono::steady_clock::now() - latest_planner_message_.timestamp;
        
        if (latest_planner_message_.valid) {
          // Valid planner message within timeout - use it
          // Update upper body control state based on this message
          has_upper_body_control_ = latest_planner_message_.upper_body_position.has_value();

          // Update hand joints control state based on this message
          has_hand_joints_ = latest_planner_message_.left_hand_joints.has_value() || 
                             latest_planner_message_.right_hand_joints.has_value();

          MovementState mode_state(
            latest_planner_message_.mode,
            latest_planner_message_.movement,
            latest_planner_message_.facing,
            latest_planner_message_.speed,
            latest_planner_message_.height
          );

          if (is_squat_motion_mode(static_cast<LocomotionMode>(mode_state.locomotion_mode))) {
            if (mode_state.height < 0.2) mode_state.height = 0.2;
          }
          if (is_static_motion_mode(static_cast<LocomotionMode>(mode_state.locomotion_mode))) {
            mode_state.movement_speed = -1.0f;
          }

          // normalize facing direction and movement direction
          mode_state.facing_direction = normalize_vector_d(mode_state.facing_direction);
          mode_state.movement_direction = normalize_vector_d(mode_state.movement_direction);

          movement_state_buffer.SetData(mode_state);
          
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[ZMQManager] Planner command: mode=" << latest_planner_message_.mode 
                      << ", speed=" << latest_planner_message_.speed << std::endl;
          }

          // Clear planner buffer to avoid using stale data
          latest_planner_message_.valid = false;

        } else if (!latest_planner_message_.valid && time_since_last_planner >= PLANNER_TIMEOUT) {
          // Planner timeout - reset to IDLE and clear buffer
          has_upper_body_control_ = false;

          has_hand_joints_ = false;

          auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
          MovementState idle_state(
            static_cast<int>(LocomotionMode::IDLE),
            {0.0f, 0.0f, 0.0f},
            current_facing,
            -1.0f,
            -1.0f
          );
          movement_state_buffer.SetData(idle_state);
          
          if (latest_planner_message_.timestamp != std::chrono::steady_clock::time_point{}) {
            std::cout << "[ZMQManager] Planner timeout (" 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(time_since_last_planner).count()
                      << "ms) - reset to IDLE and cleared buffer" << std::endl;

            // Clear planner buffer to avoid using stale data
            latest_planner_message_.valid = false;
            latest_planner_message_.timestamp = {};
          }
          
        }
      }

      if (has_vr_3point_control_ && !last_has_vr_3point_control_) {
        std::cout << "[ZMQManager] VR 3-point control enabled" << std::endl;
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        if (current_motion->GetEncodeMode() >= 0) {
              current_motion->SetEncodeMode(1);
        }
      }
      else if (!has_vr_3point_control_ && last_has_vr_3point_control_) {
        std::cout << "[ZMQManager] VR 3-point control disabled" << std::endl;
        std::lock_guard<std::mutex> lock(current_motion_mutex);
        if (current_motion->GetEncodeMode() >= 0) {
              current_motion->SetEncodeMode(0);
        }
      }
      last_has_vr_3point_control_ = has_vr_3point_control_;
    }

    // Callback handlers - just update buffer, no queue
    void OnCommandReceived(
        const std::string& topic,
        const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
        const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
      
      if (hdr.fields.empty() || bufs.empty()) return;
      
      int start_idx = -1, stop_idx = -1, planner_idx = -1;
      for (size_t i = 0; i < hdr.fields.size(); ++i) {
        if (hdr.fields[i].name == "start") start_idx = static_cast<int>(i);
        else if (hdr.fields[i].name == "stop") stop_idx = static_cast<int>(i);
        else if (hdr.fields[i].name == "planner") planner_idx = static_cast<int>(i);
      }
      
      if (start_idx < 0 || stop_idx < 0 || planner_idx < 0) {
        std::cerr << "[ZMQManager] Command missing fields (need: start, stop, planner)" << std::endl;
        return;
      }
      
      CommandMessage cmd;
      cmd.valid = true;
      
      bool needs_swap = hdr.NeedsByteSwap();
      
      // Decode start
      const auto& start_buf = bufs[start_idx];
      const auto& start_field = hdr.fields[start_idx];
      if (start_field.dtype == "bool" || start_field.dtype == "u8") {
        uint8_t val = 0;
        if (start_buf.size >= sizeof(uint8_t)) {
          std::memcpy(&val, start_buf.data, sizeof(uint8_t));
          cmd.start = (val != 0);
        }
      } else if (start_field.dtype == "i32") {
        int32_t val = 0;
        if (start_buf.size >= sizeof(int32_t)) {
          std::memcpy(&val, start_buf.data, sizeof(int32_t));
          if (needs_swap) val = byte_swap(val);
          cmd.start = (val != 0);
        }
      }
      
      // Decode stop
      const auto& stop_buf = bufs[stop_idx];
      const auto& stop_field = hdr.fields[stop_idx];
      if (stop_field.dtype == "bool" || stop_field.dtype == "u8") {
        uint8_t val = 0;
        if (stop_buf.size >= sizeof(uint8_t)) {
          std::memcpy(&val, stop_buf.data, sizeof(uint8_t));
          cmd.stop = (val != 0);
        }
      } else if (stop_field.dtype == "i32") {
        int32_t val = 0;
        if (stop_buf.size >= sizeof(int32_t)) {
          std::memcpy(&val, stop_buf.data, sizeof(int32_t));
          if (needs_swap) val = byte_swap(val);
          cmd.stop = (val != 0);
        }
      }
      
      // Decode planner
      const auto& planner_buf = bufs[planner_idx];
      const auto& planner_field = hdr.fields[planner_idx];
      if (planner_field.dtype == "bool" || planner_field.dtype == "u8") {
        uint8_t val = 0;
        if (planner_buf.size >= sizeof(uint8_t)) {
          std::memcpy(&val, planner_buf.data, sizeof(uint8_t));
          cmd.planner = (val != 0);
        }
      } else if (planner_field.dtype == "i32") {
        int32_t val = 0;
        if (planner_buf.size >= sizeof(int32_t)) {
          std::memcpy(&val, planner_buf.data, sizeof(int32_t));
          if (needs_swap) val = byte_swap(val);
          cmd.planner = (val != 0);
        }
      }
      
      // Update buffer with OR logic to accumulate start/stop signals
      std::lock_guard<std::mutex> lock(command_mutex_);
      
      // If starting new accumulation cycle, reset start/stop
      if (!latest_command_.valid) {
        latest_command_.start = false;
        latest_command_.stop = false;
      }
      
      // Accumulate start/stop with OR logic
      latest_command_.start = latest_command_.start || cmd.start;
      latest_command_.stop = latest_command_.stop || cmd.stop;
      latest_command_.planner = cmd.planner;  // Overwrite (mode should be latest)
      latest_command_.valid = true;
      
      if constexpr (DEBUG_LOGGING) {
        std::cout << "[ZMQManager] Command received: start=" << cmd.start 
                  << ", stop=" << cmd.stop << ", planner=" << cmd.planner << std::endl;
      }
    }
    
    void OnPlannerReceived(
        const std::string& topic,
        const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
        const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
      
      int mode_idx = -1, movement_idx = -1, facing_idx = -1;
      int speed_idx = -1, height_idx = -1;
      int upper_body_position_idx = -1, upper_body_velocity_idx = -1;
      int left_hand_joints_idx = -1, right_hand_joints_idx = -1;
      int vr_position_idx = -1, vr_orientation_idx = -1, vr_compliance_idx = -1;

      for (size_t i = 0; i < hdr.fields.size(); ++i) {
        const auto& f = hdr.fields[i];
        if (f.name == "mode") mode_idx = static_cast<int>(i);
        else if (f.name == "movement") movement_idx = static_cast<int>(i);
        else if (f.name == "facing") facing_idx = static_cast<int>(i);
        else if (f.name == "speed") speed_idx = static_cast<int>(i);
        else if (f.name == "height") height_idx = static_cast<int>(i);
        else if (f.name == "upper_body_position") upper_body_position_idx = static_cast<int>(i);
        else if (f.name == "upper_body_velocity") upper_body_velocity_idx = static_cast<int>(i);
        else if (f.name == "left_hand_joints") left_hand_joints_idx = static_cast<int>(i);
        else if (f.name == "right_hand_joints") right_hand_joints_idx = static_cast<int>(i);
        else if (f.name == "vr_position") vr_position_idx = static_cast<int>(i);
        else if (f.name == "vr_orientation") vr_orientation_idx = static_cast<int>(i);
        else if (f.name == "vr_compliance") vr_compliance_idx = static_cast<int>(i);
      }
      
      if (mode_idx < 0 || movement_idx < 0 || facing_idx < 0) {
        std::cerr << "[ZMQManager] Planner missing required fields" << std::endl;
        return;
      }
      
      PlannerMessage msg;
      msg.valid = true;
      
      bool needs_swap = hdr.NeedsByteSwap();
      
      // Decode mode
      const auto& mode_buf = bufs[mode_idx];
      int32_t mode_val;
      std::memcpy(&mode_val, mode_buf.data, sizeof(int32_t));
      if (needs_swap) mode_val = byte_swap(mode_val);
      msg.mode = static_cast<int>(mode_val);
      
      // Decode movement based on dtype
      const auto& movement_buf = bufs[movement_idx];
      const auto& movement_field = hdr.fields[movement_idx];
      if (movement_field.dtype == "f32") {
        for (int i = 0; i < 3; ++i) {
          float val;
          std::memcpy(&val, static_cast<const uint8_t*>(movement_buf.data) + i * sizeof(float), sizeof(float));
          if (needs_swap) val = byte_swap(val);
          msg.movement[i] = static_cast<double>(val);
        }
      } else { // f64 or default
        for (int i = 0; i < 3; ++i) {
          double val;
          std::memcpy(&val, static_cast<const uint8_t*>(movement_buf.data) + i * sizeof(double), sizeof(double));
          if (needs_swap) val = byte_swap(val);
          msg.movement[i] = val;
        }
      }
      
      // Decode facing based on dtype
      const auto& facing_buf = bufs[facing_idx];
      const auto& facing_field = hdr.fields[facing_idx];
      if (facing_field.dtype == "f32") {
        for (int i = 0; i < 3; ++i) {
          float val;
          std::memcpy(&val, static_cast<const uint8_t*>(facing_buf.data) + i * sizeof(float), sizeof(float));
          if (needs_swap) val = byte_swap(val);
          msg.facing[i] = static_cast<double>(val);
        }
      } else { // f64 or default
        for (int i = 0; i < 3; ++i) {
          double val;
          std::memcpy(&val, static_cast<const uint8_t*>(facing_buf.data) + i * sizeof(double), sizeof(double));
          if (needs_swap) val = byte_swap(val);
          msg.facing[i] = val;
        }
      }
      
      // Optional: speed (decode based on dtype)
      if (speed_idx >= 0) {
        const auto& speed_buf = bufs[speed_idx];
        const auto& speed_field = hdr.fields[speed_idx];
        if (speed_field.dtype == "f32") {
          float val;
          std::memcpy(&val, speed_buf.data, sizeof(float));
          if (needs_swap) val = byte_swap(val);
          msg.speed = static_cast<double>(val);
        } else { // f64 or default
          double val;
          std::memcpy(&val, speed_buf.data, sizeof(double));
          if (needs_swap) val = byte_swap(val);
          msg.speed = val;
        }
      }
      
      // Optional: height (decode based on dtype)
      if (height_idx >= 0) {
        const auto& height_buf = bufs[height_idx];
        const auto& height_field = hdr.fields[height_idx];
        if (height_field.dtype == "f32") {
          float val;
          std::memcpy(&val, height_buf.data, sizeof(float));
          if (needs_swap) val = byte_swap(val);
          msg.height = static_cast<double>(val);
        } else { // f64 or default
          double val;
          std::memcpy(&val, height_buf.data, sizeof(double));
          if (needs_swap) val = byte_swap(val);
          msg.height = val;
        }
      }

      // Optional: upper_body_position (17 DOF, decode based on dtype)
      if (upper_body_position_idx >= 0) {
        const auto& ub_pos_buf = bufs[upper_body_position_idx];
        const auto& ub_pos_field = hdr.fields[upper_body_position_idx];

        std::array<double, 17> upper_body_position_data{};
        if (ub_pos_field.dtype == "f32") {
          for (int i = 0; i < 17; ++i) {
            float val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(ub_pos_buf.data) + i * sizeof(float),
                        sizeof(float));
            if (needs_swap) val = byte_swap(val);
            upper_body_position_data[i] = static_cast<double>(val);
          }
        } else { // f64 or default
          for (int i = 0; i < 17; ++i) {
            double val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(ub_pos_buf.data) + i * sizeof(double),
                        sizeof(double));
            if (needs_swap) val = byte_swap(val);
            upper_body_position_data[i] = val;
          }
        }
        msg.upper_body_position = upper_body_position_data;

        // Push into upper-body position buffer
        upper_body_joint_positions_.SetData(upper_body_position_data);
      }

      // Optional: upper_body_velocity (17 DOF, decode based on dtype)
      if (upper_body_velocity_idx >= 0) {
        const auto& ub_vel_buf = bufs[upper_body_velocity_idx];
        const auto& ub_vel_field = hdr.fields[upper_body_velocity_idx];

        std::array<double, 17> upper_body_velocity_data{};
        if (ub_vel_field.dtype == "f32") {
          for (int i = 0; i < 17; ++i) {
            float val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(ub_vel_buf.data) + i * sizeof(float),
                        sizeof(float));
            if (needs_swap) val = byte_swap(val);
            upper_body_velocity_data[i] = static_cast<double>(val);
          }
        } else { // f64 or default
          for (int i = 0; i < 17; ++i) {
            double val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(ub_vel_buf.data) + i * sizeof(double),
                        sizeof(double));
            if (needs_swap) val = byte_swap(val);
            upper_body_velocity_data[i] = val;
          }
        }
        msg.upper_body_velocity = upper_body_velocity_data;

        // Push into upper-body velocity buffer
        upper_body_joint_velocities_.SetData(upper_body_velocity_data);
      }
      
      // Optional: left_hand_joints (7 DOF, decode based on dtype)
      if (left_hand_joints_idx >= 0) {
        const auto& lh_buf = bufs[left_hand_joints_idx];
        const auto& lh_field = hdr.fields[left_hand_joints_idx];

        std::array<double, 7> left_hand_joints_data{};
        if (lh_field.dtype == "f32") {
          for (int i = 0; i < 7; ++i) {
            float val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(lh_buf.data) + i * sizeof(float),
                        sizeof(float));
            if (needs_swap) val = byte_swap(val);
            left_hand_joints_data[i] = static_cast<double>(val);
          }
        } else { // f64 or default
          for (int i = 0; i < 7; ++i) {
            double val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(lh_buf.data) + i * sizeof(double),
                        sizeof(double));
            if (needs_swap) val = byte_swap(val);
            left_hand_joints_data[i] = val;
          }
        }
        msg.left_hand_joints = left_hand_joints_data;

        // Push into left hand joint buffer
        left_hand_joint_.SetData(left_hand_joints_data);
      }

      // Optional: right_hand_joints (7 DOF, decode based on dtype)
      if (right_hand_joints_idx >= 0) {
        const auto& rh_buf = bufs[right_hand_joints_idx];
        const auto& rh_field = hdr.fields[right_hand_joints_idx];

        std::array<double, 7> right_hand_joints_data{};
        if (rh_field.dtype == "f32") {
          for (int i = 0; i < 7; ++i) {
            float val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(rh_buf.data) + i * sizeof(float),
                        sizeof(float));
            if (needs_swap) val = byte_swap(val);
            right_hand_joints_data[i] = static_cast<double>(val);
          }
        } else { // f64 or default
          for (int i = 0; i < 7; ++i) {
            double val;
            std::memcpy(&val,
                        static_cast<const uint8_t*>(rh_buf.data) + i * sizeof(double),
                        sizeof(double));
            if (needs_swap) val = byte_swap(val);
            right_hand_joints_data[i] = val;
          }
        }
        msg.right_hand_joints = right_hand_joints_data;

        // Push into right hand joint buffer
        right_hand_joint_.SetData(right_hand_joints_data);
      }

      // Decode VR 3-point tracking data if present (9 doubles for position, 12 doubles for orientation, 3 doubles for compliance)
      // Use default values from InputInterface as fallback
      bool has_vr_position = (vr_position_idx >= 0);
      bool has_vr_orientation = (vr_orientation_idx >= 0);
      bool has_vr_compliance = (vr_compliance_idx >= 0);
      // Default values from input_interface.hpp
      std::array<double, 9> vr_position_values = {0.0903,  0.1615, -0.2411,   // left wrist xyz
                                                    0.1280, -0.1522, -0.2461,   // right wrist xyz
                                                    0.0241, -0.0081,  0.4028};  // head xyz
      std::array<double, 12> vr_orientation_values = {0.7295,  0.3145,  0.5533, -0.2506,   // left quat (w,x,y,z)
                                                        0.7320, -0.2639,  0.5395,  0.3217,   // right quat (w,x,y,z)
                                                        0.9991,  0.011,  0.0402, -0.0002};  // head quat (w,x,y,z)
      std::array<double, 3> vr_compliance_values = GetVR3PointCompliance();
      
      if (has_vr_position) {
          const auto& vr_pos_field = hdr.fields[vr_position_idx];
          const auto& vr_pos_buf = bufs[vr_position_idx];
          
          // Validate shape: expect [9] or [1, 9]
          int num_vr_pos_values = 0;
          if (vr_pos_field.shape.size() == 1 && vr_pos_field.shape[0] == 9) {
              num_vr_pos_values = 9;
          } else if (vr_pos_field.shape.size() == 2 && vr_pos_field.shape[1] == 9) {
              num_vr_pos_values = 9;
          }
          
          if (num_vr_pos_values == 9) {
              // Decode 9 position values
              if (vr_pos_field.dtype == "f32") {
                  for (int j = 0; j < 9; ++j) {
                      float val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_pos_buf.data) + j * sizeof(float), sizeof(float));
                      if (needs_swap) val = byte_swap(val);
                      vr_position_values[j] = static_cast<double>(val);
                  }
              } else { // f64 or default
                  for (int j = 0; j < 9; ++j) {
                      double val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_pos_buf.data) + j * sizeof(double), sizeof(double));
                      if (needs_swap) val = byte_swap(val);
                      vr_position_values[j] = val;
                  }
              }

              if constexpr (DEBUG_LOGGING) {
                  std::cout << "[ZMQManager] Decoded vr_position: [";
                  for (int j = 0; j < 9; ++j) {
                      if (j > 0) std::cout << ", ";
                      std::cout << std::fixed << std::setprecision(4) << vr_position_values[j];
                  }
                  std::cout << "]" << std::endl;
              }
          } else {
              std::cerr << "[ZMQManager] Invalid vr_position shape" << std::endl;
              has_vr_position = false;
          }
      }
      
      if (has_vr_orientation) {
          const auto& vr_orient_field = hdr.fields[vr_orientation_idx];
          const auto& vr_orient_buf = bufs[vr_orientation_idx];
          
          // Validate shape: expect [12] or [1, 12]
          int num_vr_orient_values = 0;
          if (vr_orient_field.shape.size() == 1 && vr_orient_field.shape[0] == 12) {
              num_vr_orient_values = 12;
          } else if (vr_orient_field.shape.size() == 2 && vr_orient_field.shape[1] == 12) {
              num_vr_orient_values = 12;
          }
          
          if (num_vr_orient_values == 12) {
              // Decode 12 orientation values (quaternions)
              if (vr_orient_field.dtype == "f32") {
                  for (int j = 0; j < 12; ++j) {
                      float val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_orient_buf.data) + j * sizeof(float), sizeof(float));
                      if (needs_swap) val = byte_swap(val);
                      vr_orientation_values[j] = static_cast<double>(val);
                  }
              } else { // f64 or default
                  for (int j = 0; j < 12; ++j) {
                      double val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_orient_buf.data) + j * sizeof(double), sizeof(double));
                      if (needs_swap) val = byte_swap(val);
                      vr_orientation_values[j] = val;
                  }
              }
              
              if constexpr (DEBUG_LOGGING) {
                  std::cout << "[ZMQManager] Decoded vr_orientation: [";
                  for (int j = 0; j < 12; ++j) {
                      if (j > 0) std::cout << ", ";
                      std::cout << std::fixed << std::setprecision(4) << vr_orientation_values[j];
                  }
                  std::cout << "]" << std::endl;
              }
          } else {
              std::cerr << "[ZMQManager] Invalid vr_orientation shape" << std::endl;
              has_vr_orientation = false;
          }
      }
      
      if (has_vr_compliance) {
          const auto& vr_compl_field = hdr.fields[vr_compliance_idx];
          const auto& vr_compl_buf = bufs[vr_compliance_idx];
          
          // Validate shape: expect [3] or [1, 3]
          int num_vr_compl_values = 0;
          if (vr_compl_field.shape.size() == 1 && vr_compl_field.shape[0] == 3) {
              num_vr_compl_values = 3;
          } else if (vr_compl_field.shape.size() == 2 && vr_compl_field.shape[1] == 3) {
              num_vr_compl_values = 3;
          }
          
          if (num_vr_compl_values == 3) {
              // Decode 3 compliance values
              if (vr_compl_field.dtype == "f32") {
                  for (int j = 0; j < 3; ++j) {
                      float val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_compl_buf.data) + j * sizeof(float), sizeof(float));
                      if (needs_swap) val = byte_swap(val);
                      vr_compliance_values[j] = static_cast<double>(val);
                  }
              } else { // f64 or default
                  for (int j = 0; j < 3; ++j) {
                      double val;
                      std::memcpy(&val, static_cast<const uint8_t*>(vr_compl_buf.data) + j * sizeof(double), sizeof(double));
                      if (needs_swap) val = byte_swap(val);
                      vr_compliance_values[j] = val;
                  }
              }
              
              if constexpr (DEBUG_LOGGING) {
                  std::cout << "[ZMQManager] Decoded vr_compliance: [";
                  for (int j = 0; j < 3; ++j) {
                      if (j > 0) std::cout << ", ";
                      std::cout << std::fixed << std::setprecision(4) << vr_compliance_values[j];
                  }
                  std::cout << "]" << std::endl;
              }
          } else {
              std::cerr << "[ZMQManager] Invalid vr_compliance shape" << std::endl;
              has_vr_compliance = false;
          }
      }

      // Handle VR 3-point tracking: vr_position is required, orientation and compliance are optional
      // If vr_position is present, set has_vr_3point_control_ = true and update all buffers
      if (has_vr_position) {
        // Always update all three buffers when VR position is present
        // (orientation and compliance will use defaults if not provided)
        vr_3point_position_.SetData(vr_position_values);
        vr_3point_orientation_.SetData(vr_orientation_values);
        if (has_vr_compliance) {
          SetVR3PointCompliance(vr_compliance_values);
        }
        has_vr_3point_control_ = true;

        pose_interface_->SetVR3PointPosition(vr_position_values);
        pose_interface_->SetVR3PointOrientation(vr_orientation_values);
        pose_interface_->SetVR3PointCompliance(vr_compliance_values);
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ZMQManager] VR 3-point data updated:" << std::endl;
            std::cout << "  Position (left, right, head): [";
            for (int j = 0; j < 9; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << vr_position_values[j];
            }
            std::cout << "]" << (has_vr_position ? " (from message)" : " (default)") << std::endl;
            
            std::cout << "  Orientation (left quat, right quat, head quat): [";
            for (int j = 0; j < 12; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4) << vr_orientation_values[j];
            }
            std::cout << "]" << (has_vr_orientation ? " (from message)" : " (default)") << std::endl;
            if (has_vr_compliance) {
              std::cout << "  Compliance (left, right, head): [";
              for (int j = 0; j < 3; ++j) {
                  if (j > 0) std::cout << ", ";
                  std::cout << std::fixed << std::setprecision(4) << vr_compliance_values[j];
              }
              std::cout << "]" << (has_vr_compliance ? " (from message)" : " (default)") << std::endl;
            }
        }
      }
      else {
        // No VR position provided - disable VR 3-point control
        has_vr_3point_control_ = false;
      }

      // Update buffer directly (no queue) and set timestamp
      std::lock_guard<std::mutex> lock(planner_mutex_);
      latest_planner_message_ = msg;
      latest_planner_message_.timestamp = std::chrono::steady_clock::now();
    }
    

  private:
    // ------------------------------------------------------------------
    // Configuration (set once in constructor)
    // ------------------------------------------------------------------
    std::string zmq_host_;           ///< ZMQ server hostname.
    int zmq_port_;                   ///< ZMQ server port.
    std::string pose_topic_;         ///< Topic for streamed motion data.
    std::string command_topic_;      ///< Topic for start / stop / mode commands.
    std::string planner_topic_;      ///< Topic for planner movement commands.
    bool zmq_conflate_;              ///< ZMQ conflate option for pose topic.
    bool zmq_verbose_;               ///< Verbose logging flag.
    
    // ------------------------------------------------------------------
    // Owned sub-components
    // ------------------------------------------------------------------
    /// Pose-streaming interface (handles STREAMED_MOTION mode internally).
    std::unique_ptr<ZMQEndpointInterface> pose_interface_;
    
    /// Background subscriber for the command topic.
    std::unique_ptr<ZMQPackedMessageSubscriber> command_subscriber_;
    /// Background subscriber for the planner topic.
    std::unique_ptr<ZMQPackedMessageSubscriber> planner_subscriber_;
    
    // ------------------------------------------------------------------
    // Mode / message state
    // ------------------------------------------------------------------
    ManagedMode active_mode_;           ///< Current operational mode (PLANNER or STREAMED_MOTION).
    
    std::mutex command_mutex_;          ///< Guards access to latest_command_.
    CommandMessage latest_command_;     ///< Most recent (or accumulated) command message.
    
    std::mutex planner_mutex_;          ///< Guards access to latest_planner_message_.
    PlannerMessage latest_planner_message_;  ///< Most recent planner movement message.
    
    // ------------------------------------------------------------------
    // Per-frame control flags (reset at start of update())
    // ------------------------------------------------------------------
    bool emergency_stop_ = false;  ///< Set by 'O'/'o' keyboard shortcut.
    bool start_control_ = false;   ///< Start request from command message.
    bool stop_control_ = false;    ///< Stop request from command message.

    /// True once the planner has been initialised and is generating motions.
    bool is_planner_ready_ = false;
    /// True when transitioning from streamed-motion (teleop) back to planner mode;
    /// used to keep forwarding VR/hand data from the pose interface until planner is ready.
    bool switch_from_teleop_to_planner_ = false;

    /// Tracks the previous frame's VR-3-point state to detect enable/disable transitions
    /// and automatically toggle encoder mode accordingly.
    bool last_has_vr_3point_control_ = false;
};

#endif // ZMQ_MANAGER_HPP
