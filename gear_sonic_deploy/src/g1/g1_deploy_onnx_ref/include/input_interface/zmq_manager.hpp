/**
 * @file zmq_manager.hpp
 * @brief Network-only input manager that switches between planner mode and
 *        streamed-motion mode, both driven by ZMQ topics.
 *
 * ZMQManager subscribes to **three** ZMQ topics on the same host:port:
 *
 *   Topic      | Purpose
 *   -----------|--------
 *   command    | High-level control (claim / start / stop / mode switch).
 *              | Protocol v2 requires receiver epoch, publisher session,
 *              | command index, claim, start, stop, and planner fields.
 *   planner    | Per-frame locomotion commands (mode, movement, facing, speed,
 *              | optional upper-body / hand / VR data) plus session tokens.
 *   pose       | Streamed motion frames (joint_pos, joint_vel, body_quat, …).
 *              | Under zmq_manager, every frame also carries session tokens.
 *
 * ## Control-session ownership
 *
 * A process-random 128-bit receiver epoch is published on the output
 * `control_session` topic. The manager waits for it, sends an explicit claim,
 * and waits for acknowledgement before publishing control data. The first
 * valid publisher session wins. Binding and replay counters persist until
 * native process exit; no stop, mode, or reconnect path resets them.
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
 * ## Active-input lease
 *
 * A start request is accepted only when the selected mode already has fresh
 * input. Once control is active, loss of planner or pose messages for 500 ms
 * requests a terminal stop. This native deadman remains effective if the
 * Python publisher is killed before it can send a stop message. Restart the
 * native deployment before starting another manager session.
 *
 * Standalone `--input-type zmq` retains legacy wire compatibility.
 * `--input-type zmq_manager` requires protocol v2 session fields and fails
 * closed. Session tokens provide replay scoping and single-publisher
 * ownership, not encryption or authentication; use only on a trusted,
 * firewall-scoped LAN.
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
#include <atomic>
#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>

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

    static constexpr auto INPUT_LEASE_TIMEOUT = std::chrono::milliseconds(500);
    static constexpr auto INPUT_LEASE_FUTURE_TOLERANCE = std::chrono::milliseconds(50);
    static constexpr auto INPUT_ACTIVATION_GRACE = std::chrono::milliseconds(250);

    static bool IsInputLeaseFreshAt(
        ManagedMode mode,
        std::optional<std::chrono::steady_clock::time_point> planner_timestamp,
        std::optional<std::chrono::steady_clock::time_point> pose_timestamp,
        std::chrono::steady_clock::time_point now) {
      const auto& selected_timestamp =
          mode == ManagedMode::PLANNER ? planner_timestamp : pose_timestamp;
      if (!selected_timestamp.has_value()) {
        return false;
      }
      const auto age = now - *selected_timestamp;
      return age >= -INPUT_LEASE_FUTURE_TOLERANCE &&
             age <= INPUT_LEASE_TIMEOUT;
    }

    static bool IsPlannerFrameAdvancing(
        std::optional<int64_t> last_frame, int64_t current_frame) {
      return current_frame >= 0 &&
             (!last_frame.has_value() || current_frame > *last_frame);
    }

    static bool IsCommandIndexAdvancing(
        std::optional<int64_t> last_index, int64_t current_index) {
      return current_index >= 0 &&
             (!last_index.has_value() || current_index > *last_index);
    }

    ZMQManager(
      const std::string& zmq_host,
      int zmq_port,
      const std::string& pose_topic = "pose",
      const std::string& command_topic = "command",
      const std::string& planner_topic = "planner",
      bool zmq_conflate = false,
      bool zmq_verbose = false,
      std::shared_ptr<ControlSessionState> control_session_state = nullptr
    ) : InputInterface(), 
        zmq_host_(zmq_host), 
        zmq_port_(zmq_port), 
        pose_topic_(pose_topic),
        command_topic_(command_topic),
        planner_topic_(planner_topic),
        zmq_conflate_(zmq_conflate), 
        zmq_verbose_(zmq_verbose),
        control_session_state_(std::move(control_session_state)) {
      
      if (!control_session_state_) {
        throw std::invalid_argument(
            "ZMQManager requires a control-session state");
      }
      type_ = InputType::NETWORK;
      active_mode_ = ManagedMode::PLANNER;  // Default to planner mode
      
      // Create pose interface (for streamed motion mode)
      pose_interface_ = std::make_unique<ZMQEndpointInterface>(
        zmq_host_, zmq_port_, pose_topic_, zmq_conflate_, zmq_verbose_,
        control_session_state_
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
      std::cout
          << "    Format v2: { receiver_epoch: u8[16], "
          << "publisher_session: u8[16], command_index: i64[1], "
          << "claim: bool[1], start: bool[1], stop: bool[1], "
          << "planner: bool[1] }" << std::endl;
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
      report_temperature_flag_ = false;
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
          case 'f':
          case 'F':
            report_temperature_flag_ = true;
            is_manager_key = true;
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
          ManagedMode new_mode = latest_command_.planner
                                     ? ManagedMode::PLANNER
                                     : ManagedMode::STREAMED_MOTION;
          // Set control flags (already accumulated in callback)
          if (latest_command_.start) {
            if (new_mode == ManagedMode::PLANNER) {
              std::lock_guard<std::mutex> planner_lock(planner_mutex_);
              latest_planner_message_.valid = false;
              latest_planner_message_.timestamp = {};
            } else if (pose_interface_) {
              pose_interface_->RequireFreshFrameAfterActivation();
            }
            start_request_pending_ = true;
            start_request_deadline_ =
                std::chrono::steady_clock::now() + INPUT_ACTIVATION_GRACE;
          }
          if (latest_command_.stop) {
            stop_control_ = true;
            start_request_pending_ = false;
            mode_transition_pending_ = false;
          }

          // Handle mode switching
          if (new_mode != active_mode_) {
            mode_transition_pending_ = true;
            mode_transition_deadline_ =
                std::chrono::steady_clock::now() + INPUT_ACTIVATION_GRACE;
            // Trigger safety reset on mode switch
            TriggerSafetyReset();
            if (pose_interface_) {
              pose_interface_->TriggerSafetyReset();
            }

            if (new_mode == ManagedMode::PLANNER) {
              std::cout << "[ZMQManager] Switched to: PLANNER mode (safety reset)" << std::endl;
              std::lock_guard<std::mutex> planner_lock(planner_mutex_);
              latest_planner_message_.valid = false;
              latest_planner_message_.timestamp = {};
              is_planner_ready_ = false;
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
                      std::mutex& current_motion_mutex,
                      bool& report_temperature) override {
      if (!has_planner) {
        std::cerr << "[ZMQCommandManager ERROR] Planner not available in planner mode" << std::endl;
        operator_state.stop = true;
        return;
      }
      // Emergency stop
      if (report_temperature_flag_) {
        report_temperature = true;
        report_temperature_flag_ = false;
      }
      if (emergency_stop_) {
        operator_state.stop = true;
        start_request_pending_ = false;
        mode_transition_pending_ = false;
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
        start_request_pending_ = false;
        mode_transition_pending_ = false;
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
        return;
      }
      if (operator_state.stop) {
        operator_state.play = false;
        start_request_pending_ = false;
        mode_transition_pending_ = false;
        planner_state.enabled = false;
        planner_state.initialized = false;
        return;
      }

      // Pose packets must be decoded before they are allowed to renew the
      // lease. This also resolves the cross-SUB ordering race where the command
      // can arrive just before the first pose packet is processed.
      if (active_mode_ == ManagedMode::STREAMED_MOTION && pose_interface_) {
        pose_interface_->handle_input(
            motion_reader,
            current_motion,
            current_frame,
            operator_state,
            reinitialize_heading,
            heading_state_buffer,
            has_planner,
            planner_state,
            movement_state_buffer,
            current_motion_mutex,
            report_temperature);
        if (operator_state.stop) {
          return;
        }
      }

      const auto now = std::chrono::steady_clock::now();
      const bool input_is_fresh = HasFreshActiveInput(now);

      // A mode switch is de-energized while the two independent SUB sockets
      // converge. The grace is bounded and never renews from malformed data.
      if (mode_transition_pending_) {
        if (!input_is_fresh) {
          operator_state.play = false;
          if (now <= mode_transition_deadline_) {
            return;
          }
          StopForExpiredInputLease(operator_state, planner_state);
          return;
        }
        mode_transition_pending_ = false;
      }

      if (operator_state.start) {
        start_request_pending_ = false;
      } else if (start_request_pending_) {
        if (!input_is_fresh) {
          operator_state.play = false;
          if (now <= start_request_deadline_) {
            return;
          }
          StopForExpiredInputLease(operator_state, planner_state);
          return;
        }
        start_request_pending_ = false;
        start_control_ = true;
      }

      // Once active, there is no ordinary dropout grace: the selected input
      // must remain inside its 500 ms lease on every control iteration.
      if (operator_state.start && !input_is_fresh) {
        StopForExpiredInputLease(operator_state, planner_state);
        return;
      }

      if (active_mode_ == ManagedMode::PLANNER) {
        handlePlannerInput(
            motion_reader,
            current_motion,
            current_frame,
            operator_state,
            reinitialize_heading,
            heading_state_buffer,
            has_planner,
            planner_state,
            movement_state_buffer,
            current_motion_mutex);
      } else if (start_control_ && !operator_state.start) {
        operator_state.start = true;
        operator_state.play = true;
        reinitialize_heading = true;
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
      if (active_mode_ == ManagedMode::PLANNER) {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        if (latest_planner_message_.timestamp !=
            std::chrono::steady_clock::time_point{}) {
          return latest_planner_message_.timestamp;
        }
        return {};
      }
      if (pose_interface_) {
        return pose_interface_->GetLastUpdateTime();
      }
      return {};
    }

    bool IsActiveInputLeaseFreshAt(
        std::chrono::steady_clock::time_point now) const {
      return HasFreshActiveInput(now);
    }

  private:
    friend struct ZMQWireTestAccess;

    bool HasFreshActiveInput(std::chrono::steady_clock::time_point now) const {
      std::optional<std::chrono::steady_clock::time_point> planner_timestamp;
      {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        if (latest_planner_message_.timestamp !=
            std::chrono::steady_clock::time_point{}) {
          planner_timestamp = latest_planner_message_.timestamp;
        }
      }
      const auto pose_timestamp =
          pose_interface_ ? pose_interface_->GetLastUpdateTime()
                          : std::optional<std::chrono::steady_clock::time_point>{};
      return IsInputLeaseFreshAt(
          active_mode_,
          planner_timestamp,
          pose_timestamp,
          now);
    }

    void StopForExpiredInputLease(
        OperatorState& operator_state,
        PlannerState& planner_state) {
      operator_state.stop = true;
      operator_state.play = false;
      planner_state.enabled = false;
      planner_state.initialized = false;
      start_request_pending_ = false;
      mode_transition_pending_ = false;
      {
        std::lock_guard<std::mutex> lock(planner_mutex_);
        latest_planner_message_.valid = false;
        latest_planner_message_.timestamp = {};
      }
      has_upper_body_control_ = false;
      has_hand_joints_ = false;
      has_vr_3point_control_ = false;
      if (pose_interface_) {
        pose_interface_->TriggerSafetyReset();
      }
      std::cerr
          << "[ZMQManager SAFETY] Active "
          << (active_mode_ == ManagedMode::PLANNER ? "planner" : "pose")
          << " input lease expired or was never established; stopping control"
          << std::endl;
    }

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
      const auto stop_if_input_lease_expired = [&]() {
        if (operator_state.stop) {
          operator_state.play = false;
          return true;
        }
        if (HasFreshActiveInput(std::chrono::steady_clock::now())) {
          return false;
        }
        StopForExpiredInputLease(operator_state, planner_state);
        return true;
      };

      // Handle safety reset from interface manager (same as GamepadManager)
      if (CheckAndClearSafetyReset()) {
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex);
          operator_state.play = false;
        }
        if (operator_state.start) {
          if (planner_state.enabled && planner_state.initialized) {
            if (stop_if_input_lease_expired()) {
              return;
            }
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
              if (stop_if_input_lease_expired()) {
                return;
              }
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                if (current_motion->name == "planner_motion") {
                  break;
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(100));
              if (stop_if_input_lease_expired()) {
                return;
              }
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

            if (stop_if_input_lease_expired()) {
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
          if (stop_if_input_lease_expired()) {
            return;
          }
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            if (current_motion->name == "planner_motion") {
              std::cout << "[ZMQManager] motion name is planner_motion" << std::endl;
              break;
            }
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          if (stop_if_input_lease_expired()) {
            return;
          }
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

        if (stop_if_input_lease_expired()) {
          return;
        }
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
          if (latest_planner_message_.upper_body_position.has_value()) {
            upper_body_joint_positions_.SetData(
                *latest_planner_message_.upper_body_position);
          }
          if (latest_planner_message_.upper_body_velocity.has_value()) {
            upper_body_joint_velocities_.SetData(
                *latest_planner_message_.upper_body_velocity);
          }
          if (latest_planner_message_.left_hand_joints.has_value()) {
            left_hand_joint_.SetData(
                *latest_planner_message_.left_hand_joints);
          }
          if (latest_planner_message_.right_hand_joints.has_value()) {
            right_hand_joint_.SetData(
                *latest_planner_message_.right_hand_joints);
          }

          has_upper_body_control_ = latest_planner_message_.upper_body_position.has_value();
          has_hand_joints_ = latest_planner_message_.left_hand_joints.has_value() || 
                             latest_planner_message_.right_hand_joints.has_value();
          if (latest_planner_message_.vr_position.has_value()) {
            vr_3point_position_.SetData(*latest_planner_message_.vr_position);
            if (latest_planner_message_.vr_orientation.has_value()) {
              vr_3point_orientation_.SetData(
                  *latest_planner_message_.vr_orientation);
            }
            if (latest_planner_message_.vr_compliance.has_value()) {
              SetVR3PointCompliance(
                  *latest_planner_message_.vr_compliance);
            }
            has_vr_3point_control_ = true;
            if (pose_interface_) {
              pose_interface_->SetVR3PointPosition(
                  *latest_planner_message_.vr_position);
              if (latest_planner_message_.vr_orientation.has_value()) {
                pose_interface_->SetVR3PointOrientation(
                    *latest_planner_message_.vr_orientation);
              }
              if (latest_planner_message_.vr_compliance.has_value()) {
                pose_interface_->SetVR3PointCompliance(
                    *latest_planner_message_.vr_compliance);
              }
            }
          } else {
            has_vr_3point_control_ = false;
          }

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

      if (hdr.version != 2 || hdr.fields.empty() ||
          bufs.size() != hdr.fields.size()) {
        std::cerr << "[ZMQManager] Command field/buffer count mismatch"
                  << std::endl;
        return;
      }

      int receiver_epoch_idx = -1, publisher_session_idx = -1;
      int command_index_idx = -1, claim_idx = -1;
      int start_idx = -1, stop_idx = -1, planner_idx = -1;
      std::unordered_set<std::string> seen_fields;
      for (size_t i = 0; i < hdr.fields.size(); ++i) {
        if (!seen_fields.insert(hdr.fields[i].name).second) {
          std::cerr << "[ZMQManager] Command contains duplicate field '"
                    << hdr.fields[i].name << "'" << std::endl;
          return;
        }
        if (hdr.fields[i].name == "receiver_epoch") {
          receiver_epoch_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "publisher_session") {
          publisher_session_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "command_index") {
          command_index_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "claim") {
          claim_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "start") {
          start_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "stop") {
          stop_idx = static_cast<int>(i);
        } else if (hdr.fields[i].name == "planner") {
          planner_idx = static_cast<int>(i);
        } else {
          std::cerr << "[ZMQManager] Command contains unknown field '"
                    << hdr.fields[i].name << "'" << std::endl;
          return;
        }
      }

      if (receiver_epoch_idx < 0 || publisher_session_idx < 0 ||
          command_index_idx < 0 || claim_idx < 0 || start_idx < 0 ||
          stop_idx < 0 || planner_idx < 0) {
        std::cerr << "[ZMQManager] Command missing control-session fields"
                  << std::endl;
        return;
      }

      const bool needs_swap = hdr.NeedsByteSwap();
      const auto decode_token =
          [&](int index, ControlSessionToken& destination) {
            if (index < 0 ||
                static_cast<std::size_t>(index) >= bufs.size()) {
              return false;
            }
            const auto& field = hdr.fields[index];
            const auto& buffer = bufs[index];
            if (field.dtype != "u8" ||
                field.shape != std::vector<std::size_t>{
                                   CONTROL_SESSION_TOKEN_SIZE} ||
                buffer.data == nullptr ||
                buffer.size != CONTROL_SESSION_TOKEN_SIZE) {
              return false;
            }
            const auto* bytes =
                static_cast<const std::uint8_t*>(buffer.data);
            const auto parsed = ControlSessionState::ParseWireToken(
                std::span<const std::uint8_t>(
                    bytes, CONTROL_SESSION_TOKEN_SIZE));
            if (!parsed.has_value()) {
              return false;
            }
            destination = *parsed;
            return true;
          };
      const auto decode_boolean_scalar = [&](int index, bool& destination) {
        if (index < 0 || static_cast<std::size_t>(index) >= bufs.size()) {
          return false;
        }
        const auto& field = hdr.fields[index];
        const auto& buffer = bufs[index];
        if (field.shape.size() != 1 || field.shape[0] != 1 ||
            buffer.data == nullptr) {
          return false;
        }
        if (field.dtype == "bool" || field.dtype == "u8") {
          if (buffer.size != sizeof(uint8_t)) {
            return false;
          }
          uint8_t value = 0;
          std::memcpy(&value, buffer.data, sizeof(value));
          if (value > 1) {
            return false;
          }
          destination = value == 1;
          return true;
        }
        if (field.dtype == "i32") {
          if (buffer.size != sizeof(int32_t)) {
            return false;
          }
          int32_t value = 0;
          std::memcpy(&value, buffer.data, sizeof(value));
          if (needs_swap) {
            value = byte_swap(value);
          }
          if (value != 0 && value != 1) {
            return false;
          }
          destination = value == 1;
          return true;
        }
        return false;
      };

      CommandMessage cmd;
      if (!decode_token(receiver_epoch_idx, cmd.receiver_epoch) ||
          !decode_token(publisher_session_idx, cmd.publisher_session) ||
          !decode_boolean_scalar(claim_idx, cmd.claim) ||
          !decode_boolean_scalar(start_idx, cmd.start) ||
          !decode_boolean_scalar(stop_idx, cmd.stop) ||
          !decode_boolean_scalar(planner_idx, cmd.planner)) {
        std::cerr
            << "[ZMQManager] Command field dtype, shape, value, or buffer size is invalid"
            << std::endl;
        return;
      }

      const auto& command_index_field = hdr.fields[command_index_idx];
      const auto& command_index_buffer = bufs[command_index_idx];
      if (command_index_field.dtype != "i64" ||
          command_index_field.shape != std::vector<std::size_t>{1} ||
          command_index_buffer.data == nullptr ||
          command_index_buffer.size != sizeof(std::int64_t)) {
        std::cerr << "[ZMQManager] Invalid command index field"
                  << std::endl;
        return;
      }
      std::memcpy(
          &cmd.command_index,
          command_index_buffer.data,
          sizeof(cmd.command_index));
      if (needs_swap) {
        cmd.command_index = byte_swap(cmd.command_index);
      }
      if (cmd.command_index < 0 || (cmd.start && cmd.stop) ||
          (cmd.claim && (cmd.start || cmd.stop)) ||
          (!cmd.claim && !cmd.start && !cmd.stop)) {
        std::cerr << "[ZMQManager] Invalid command action"
                  << std::endl;
        return;
      }

      // STOP is terminal and fail-safe. Any publisher that knows the current
      // receiver epoch may stop, but it cannot change mode, claim ownership, or
      // start control. Binding and replay counters remain intact until process
      // exit.
      if (cmd.stop) {
        if (!control_session_state_->IsReceiverEpoch(
                cmd.receiver_epoch)) {
          std::cerr << "[ZMQManager] STOP receiver epoch mismatch"
                    << std::endl;
          return;
        }
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (!latest_command_.valid) {
          latest_command_.start = false;
          latest_command_.stop = false;
        }
        latest_command_.stop = true;
        latest_command_.planner =
            active_mode_ == ManagedMode::PLANNER;
        latest_command_.valid = true;
        return;
      }

      if (cmd.claim) {
        const auto result = control_session_state_->ClaimPublisher(
            cmd.receiver_epoch, cmd.publisher_session);
        if (result != ControlSessionState::ClaimResult::CLAIMED &&
            result !=
                ControlSessionState::ClaimResult::ALREADY_CLAIMED) {
          std::cerr << "[ZMQManager] Publisher claim rejected"
                    << std::endl;
          return;
        }
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (last_accepted_command_index_.has_value() &&
            cmd.command_index < *last_accepted_command_index_) {
          std::cerr << "[ZMQManager] Claim command index regressed"
                    << std::endl;
          return;
        }
        if (!last_accepted_command_index_.has_value() ||
            cmd.command_index > *last_accepted_command_index_) {
          last_accepted_command_index_ = cmd.command_index;
        }
        return;
      }

      if (!control_session_state_->IsClaimedPublisher(
              cmd.receiver_epoch, cmd.publisher_session)) {
        std::cerr << "[ZMQManager] Command publisher is not owner"
                  << std::endl;
        return;
      }

      // Update buffer with OR logic to accumulate start/stop signals
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (!IsCommandIndexAdvancing(
              last_accepted_command_index_, cmd.command_index)) {
        std::cerr << "[ZMQManager] Command index did not advance"
                  << std::endl;
        return;
      }
      last_accepted_command_index_ = cmd.command_index;
      cmd.valid = true;

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
      if (hdr.version != 2) {
        std::cerr << "[ZMQManager] Planner requires session protocol v2"
                  << std::endl;
        return;
      }
      int receiver_epoch_idx = -1, publisher_session_idx = -1;
      int frame_index_idx = -1, mode_idx = -1, movement_idx = -1,
          facing_idx = -1;
      int speed_idx = -1, height_idx = -1;
      int upper_body_position_idx = -1, upper_body_velocity_idx = -1;
      int left_hand_joints_idx = -1, right_hand_joints_idx = -1;
      int vr_position_idx = -1, vr_orientation_idx = -1, vr_compliance_idx = -1;
      std::unordered_set<std::string> seen_fields;

      for (size_t i = 0; i < hdr.fields.size(); ++i) {
        const auto& f = hdr.fields[i];
        if (!seen_fields.insert(f.name).second) {
          std::cerr << "[ZMQManager] Planner contains duplicate field '"
                    << f.name << "'" << std::endl;
          return;
        }
        if (f.name == "receiver_epoch") {
          receiver_epoch_idx = static_cast<int>(i);
        } else if (f.name == "publisher_session") {
          publisher_session_idx = static_cast<int>(i);
        } else if (f.name == "frame_index") frame_index_idx = static_cast<int>(i);
        else if (f.name == "mode") mode_idx = static_cast<int>(i);
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
        else {
          std::cerr << "[ZMQManager] Planner contains unknown field '"
                    << f.name << "'" << std::endl;
          return;
        }
      }
      
      if (receiver_epoch_idx < 0 || publisher_session_idx < 0 ||
          frame_index_idx < 0 || mode_idx < 0 || movement_idx < 0 ||
          facing_idx < 0) {
        std::cerr << "[ZMQManager] Planner missing required fields" << std::endl;
        return;
      }

      if (bufs.size() != hdr.fields.size()) {
        std::cerr << "[ZMQManager] Planner field/buffer count mismatch" << std::endl;
        return;
      }

      const auto valid_field =
          [&](int index,
              std::size_t expected_elements,
              std::initializer_list<const char*> allowed_dtypes) {
            if (index < 0 || static_cast<std::size_t>(index) >= bufs.size()) {
              return false;
            }
            const auto& field = hdr.fields[index];
            const auto& buffer = bufs[index];
            const bool dtype_allowed = std::any_of(
                allowed_dtypes.begin(),
                allowed_dtypes.end(),
                [&](const char* dtype) { return field.dtype == dtype; });
            if (!dtype_allowed || field.shape.empty() || buffer.data == nullptr) {
              return false;
            }
            std::size_t element_count = 1;
            for (const auto dimension : field.shape) {
              if (dimension == 0 ||
                  element_count >
                      std::numeric_limits<std::size_t>::max() / dimension) {
                return false;
              }
              element_count *= dimension;
            }
            const bool shape_valid =
                field.shape == std::vector<std::size_t>{expected_elements} ||
                (expected_elements > 1 &&
                 field.shape ==
                     std::vector<std::size_t>{1, expected_elements});
            return shape_valid && element_count == expected_elements &&
                   buffer.size == expected_elements * field.GetElementSize();
          };

      const auto optional_field_valid =
          [&](int index,
              std::size_t expected_elements,
              std::initializer_list<const char*> allowed_dtypes) {
            return index < 0 ||
                   valid_field(index, expected_elements, allowed_dtypes);
          };

      const auto exact_token_shape = [&](int index) {
        return index >= 0 &&
               static_cast<std::size_t>(index) < hdr.fields.size() &&
               hdr.fields[index].shape ==
                   std::vector<std::size_t>{
                       CONTROL_SESSION_TOKEN_SIZE};
      };

      if (!exact_token_shape(receiver_epoch_idx) ||
          !exact_token_shape(publisher_session_idx) ||
          !valid_field(
              receiver_epoch_idx, CONTROL_SESSION_TOKEN_SIZE, {"u8"}) ||
          !valid_field(
              publisher_session_idx, CONTROL_SESSION_TOKEN_SIZE, {"u8"}) ||
          !valid_field(frame_index_idx, 1, {"i64"}) ||
          !valid_field(mode_idx, 1, {"i32"}) ||
          !valid_field(movement_idx, 3, {"f32", "f64"}) ||
          !valid_field(facing_idx, 3, {"f32", "f64"}) ||
          !optional_field_valid(speed_idx, 1, {"f32", "f64"}) ||
          !optional_field_valid(height_idx, 1, {"f32", "f64"}) ||
          !optional_field_valid(
              upper_body_position_idx, 17, {"f32", "f64"}) ||
          !optional_field_valid(
              upper_body_velocity_idx, 17, {"f32", "f64"}) ||
          !optional_field_valid(left_hand_joints_idx, 7, {"f32", "f64"}) ||
          !optional_field_valid(right_hand_joints_idx, 7, {"f32", "f64"}) ||
          !optional_field_valid(vr_position_idx, 9, {"f32", "f64"}) ||
          !optional_field_valid(vr_orientation_idx, 12, {"f32", "f64"}) ||
          !optional_field_valid(vr_compliance_idx, 3, {"f32", "f64"})) {
        std::cerr
            << "[ZMQManager] Planner field dtype, shape, or buffer size is invalid"
            << std::endl;
        return;
      }
      
      PlannerMessage msg;
      msg.valid = true;

      const auto parse_token =
          [&](int index, ControlSessionToken& destination) {
            const auto* bytes =
                static_cast<const std::uint8_t*>(bufs[index].data);
            const auto parsed = ControlSessionState::ParseWireToken(
                std::span<const std::uint8_t>(
                    bytes, CONTROL_SESSION_TOKEN_SIZE));
            if (!parsed.has_value()) {
              return false;
            }
            destination = *parsed;
            return true;
          };
      if (!parse_token(receiver_epoch_idx, msg.receiver_epoch) ||
          !parse_token(
              publisher_session_idx, msg.publisher_session)) {
        std::cerr << "[ZMQManager] Invalid planner session token"
                  << std::endl;
        return;
      }
      
      bool needs_swap = hdr.NeedsByteSwap();

      const auto& frame_index_buf = bufs[frame_index_idx];
      int64_t frame_index_value = -1;
      std::memcpy(
          &frame_index_value, frame_index_buf.data, sizeof(frame_index_value));
      if (needs_swap) frame_index_value = byte_swap(frame_index_value);
      if (frame_index_value < 0) {
        std::cerr << "[ZMQManager] Planner frame index is negative"
                  << std::endl;
        return;
      }
      msg.frame_index = frame_index_value;
      
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
      }

      // Decode VR data transactionally. The receive thread never reads control
      // buffers owned by the input/control thread.
      bool has_vr_position = (vr_position_idx >= 0);
      bool has_vr_orientation = (vr_orientation_idx >= 0);
      bool has_vr_compliance = (vr_compliance_idx >= 0);
      if (!has_vr_position &&
          (has_vr_orientation || has_vr_compliance)) {
        std::cerr
            << "[ZMQManager] VR orientation/compliance requires position"
            << std::endl;
        return;
      }
      std::array<double, 9> vr_position_values{};
      std::array<double, 12> vr_orientation_values{};
      std::array<double, 3> vr_compliance_values{};
      
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

      const auto all_finite = [](const auto& values) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](double value) { return std::isfinite(value); });
      };
      const auto optional_finite = [&](const auto& values) {
        return !values.has_value() || all_finite(*values);
      };

      if (msg.mode < static_cast<int>(LocomotionMode::IDLE) ||
          msg.mode > static_cast<int>(LocomotionMode::SCARE_WALK) ||
          !all_finite(msg.movement) || !all_finite(msg.facing) ||
          !std::isfinite(msg.speed) || !std::isfinite(msg.height) ||
          msg.speed < -1.0 ||
          msg.speed > 10.0 || msg.height < -1.0 || msg.height > 2.0 ||
          !optional_finite(msg.upper_body_position) ||
          !optional_finite(msg.upper_body_velocity) ||
          !optional_finite(msg.left_hand_joints) ||
          !optional_finite(msg.right_hand_joints)) {
        std::cerr << "[ZMQManager] Planner values are non-finite or out of range"
                  << std::endl;
        return;
      }

      const double movement_norm = std::hypot(
          msg.movement[0], msg.movement[1], msg.movement[2]);
      const double facing_norm =
          std::hypot(msg.facing[0], msg.facing[1], msg.facing[2]);
      if (movement_norm > 1.5 || facing_norm < 1e-6 ||
          facing_norm > 1.5) {
        std::cerr
            << "[ZMQManager] Planner direction magnitude is out of range"
            << std::endl;
        return;
      }

      if (msg.upper_body_position.has_value() &&
          std::any_of(
              msg.upper_body_position->begin(),
              msg.upper_body_position->end(),
              [](double value) { return std::abs(value) > 4.0 * M_PI; })) {
        std::cerr << "[ZMQManager] Upper-body position is out of range"
                  << std::endl;
        return;
      }
      if (msg.upper_body_velocity.has_value() &&
          std::any_of(
              msg.upper_body_velocity->begin(),
              msg.upper_body_velocity->end(),
              [](double value) { return std::abs(value) > 50.0; })) {
        std::cerr << "[ZMQManager] Upper-body velocity is out of range"
                  << std::endl;
        return;
      }
      const auto hand_out_of_range = [](const auto& values) {
        return values.has_value() &&
               std::any_of(
                   values->begin(),
                   values->end(),
                   [](double value) { return std::abs(value) > 4.0 * M_PI; });
      };
      if (hand_out_of_range(msg.left_hand_joints) ||
          hand_out_of_range(msg.right_hand_joints)) {
        std::cerr << "[ZMQManager] Hand-joint position is out of range"
                  << std::endl;
        return;
      }

      if (has_vr_position) {
        if (!all_finite(vr_position_values) ||
            std::any_of(
                vr_position_values.begin(),
                vr_position_values.end(),
                [](double value) { return std::abs(value) > 10.0; })) {
          std::cerr << "[ZMQManager] VR 3-point values are invalid"
                    << std::endl;
          return;
        }
        if (has_vr_orientation) {
          if (!all_finite(vr_orientation_values)) {
            std::cerr << "[ZMQManager] VR orientation is non-finite"
                      << std::endl;
            return;
          }
          for (std::size_t quaternion = 0; quaternion < 3; ++quaternion) {
            const std::size_t offset = quaternion * 4;
            const double norm = std::hypot(
                std::hypot(
                    vr_orientation_values[offset],
                    vr_orientation_values[offset + 1]),
                std::hypot(
                    vr_orientation_values[offset + 2],
                    vr_orientation_values[offset + 3]));
            if (norm < 0.5 || norm > 1.5) {
              std::cerr << "[ZMQManager] VR quaternion is invalid"
                        << std::endl;
              return;
            }
          }
        }
        if (has_vr_compliance &&
            (!all_finite(vr_compliance_values) ||
             std::any_of(
                 vr_compliance_values.begin(),
                 vr_compliance_values.end(),
                 [](double value) {
                   return value < 0.0 || value > 0.5;
                 }))) {
          std::cerr << "[ZMQManager] VR compliance is out of range"
                    << std::endl;
          return;
        }
        msg.vr_position = vr_position_values;
        if (has_vr_orientation) {
          msg.vr_orientation = vr_orientation_values;
        }
        if (has_vr_compliance) {
          msg.vr_compliance = vr_compliance_values;
        }
      }

      // Commit one fully validated planner frame. No safety lease or target
      // buffer is updated for a partial, malformed, or non-finite packet.
      std::lock_guard<std::mutex> lock(planner_mutex_);
      if (!control_session_state_->IsClaimedPublisher(
              msg.receiver_epoch, msg.publisher_session)) {
        std::cerr << "[ZMQManager] Planner publisher is not owner"
                  << std::endl;
        return;
      }
      if (!IsPlannerFrameAdvancing(
              last_accepted_planner_frame_, msg.frame_index)) {
        std::cerr << "[ZMQManager] Planner frame did not advance"
                  << std::endl;
        return;
      }
      last_accepted_planner_frame_ = msg.frame_index;
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
    std::shared_ptr<ControlSessionState> control_session_state_;
    
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
    std::atomic<ManagedMode> active_mode_{
        ManagedMode::PLANNER};  ///< Current operational mode.
    
    std::mutex command_mutex_;          ///< Guards access to latest_command_.
    CommandMessage latest_command_;     ///< Most recent (or accumulated) command message.
    std::optional<int64_t> last_accepted_command_index_;
    
    mutable std::mutex planner_mutex_;  ///< Guards access to latest_planner_message_.
    PlannerMessage latest_planner_message_;  ///< Most recent planner movement message.
    std::optional<int64_t> last_accepted_planner_frame_;
    
    // ------------------------------------------------------------------
    // Per-frame control flags (reset at start of update())
    // ------------------------------------------------------------------
    bool emergency_stop_ = false;  ///< Set by 'O'/'o' keyboard shortcut.
    bool report_temperature_flag_ = false;  ///< Set by 'F'/'f' keyboard shortcut.
    bool start_control_ = false;   ///< Start request from command message.
    bool stop_control_ = false;    ///< Stop request from command message.
    bool start_request_pending_ = false;
    std::chrono::steady_clock::time_point start_request_deadline_{};
    bool mode_transition_pending_ = false;
    std::chrono::steady_clock::time_point mode_transition_deadline_{};

    /// True once the planner has been initialised and is generating motions.
    std::atomic<bool> is_planner_ready_{false};
    /// True when transitioning from streamed-motion (teleop) back to planner mode;
    /// used to keep forwarding VR/hand data from the pose interface until planner is ready.
    std::atomic<bool> switch_from_teleop_to_planner_{false};

    /// Tracks the previous frame's VR-3-point state to detect enable/disable transitions
    /// and automatically toggle encoder mode accordingly.
    bool last_has_vr_3point_control_ = false;
};

#endif // ZMQ_MANAGER_HPP
