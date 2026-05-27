/**
 * @file gamepad_manager.hpp
 * @brief Planner-centric gamepad manager with ZMQ delegate and D-pad motion-set selection.
 *
 * GamepadManager is a specialised InputInterface that:
 *  - Reads the Unitree wireless gamepad **directly** (no nested Gamepad class).
 *  - Operates **exclusively in planner mode** – the planner must always be loaded.
 *  - Can delegate to ZMQEndpointInterface via F1 button toggle.
 *  - D-pad selects one of 4 motion sets; face buttons cycle within the set.
 *  - Boxing set uses direct key selection instead of cycling.
 *
 * ## Gamepad Button Mapping (Planner Mode)
 *
 *   Button  | Action
 *   --------|-------
 *   Start   | Start control (enable planner, wait for init, auto-play)
 *   Select  | Emergency stop
 *   A       | Emergency stop (immediate halt)
 *   D-Up    | Select Standing motion set (SLOW_WALK default)
 *   D-Down  | Select Squat/Crawl motion set (IDEL_SQUAT default)
 *   D-Left  | Select Boxing motion set (IDEL_BOXING default)
 *   D-Right | Select Styled Walking motion set (LEDGE_WALKING default)
 *   F1      | Toggle ZMQ streaming
 *
 *   --- Standing / Squat / Styled sets (loop cycling) ---
 *   X       | Next mode in current set (wraps)
 *   Y       | Previous mode in current set (wraps)
 *   B       | Reset to set's default mode
 *   L1/R1   | Facing angle ±π/4
 *   L2/R2   | Height ±0.1 (Squat set only; disabled otherwise)
 *
 *   --- Boxing set (direct selection) ---
 *   X       | WALK_BOXING
 *   Y       | RANDOM_PUNCH
 *   B       | IDEL_BOXING (reset)
 *   L1      | LEFT_PUNCH
 *   R1      | RIGHT_PUNCH
 *   L2      | LEFT_HOOK
 *   R2      | RIGHT_HOOK
 *
 *   L stick | Movement direction (binned to nearest 30° increment)
 *   R stick | Facing direction (continuous)
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class GamepadManager
 * @brief Planner-only gamepad controller with ZMQ delegate and D-pad motion-set selection.
 *
 * When in GAMEPAD mode, button presses and stick positions are translated
 * directly into MovementState commands for the locomotion planner.
 * D-pad selects one of 4 motion sets; face buttons cycle within the set
 * (except Boxing which uses direct key selection).
 * When in ZMQ mode (toggled via F1), all calls are forwarded to ZMQ.
 */
class GamepadManager : public InputInterface {
  public:
    // ========================================
    // DEBUG CONTROL FLAG
    // ========================================
    static constexpr bool DEBUG_LOGGING = true;

    enum class ManagedType {
      GAMEPAD = 0,
      ZMQ = 1
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

      // Initialize with Standing motion set, SLOW_WALK as default
      motion_set_index_ = 0;
      current_motion_set_ = get_motion_set(motion_set_index_);
      mode_index_in_set_ = 0;
      applyModeFromSet();
    }

    void update() override {
      // Reset per-frame flags
      emergency_stop_ = false;
      report_temperature_flag_ = false;
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
          case 'o':
          case 'O':
            emergency_stop_ = true;
            is_manager_key = true;
            std::cout << "[GamepadManager] EMERGENCY STOP triggered (O/o key pressed)" << std::endl;
            break;
          case 'f':
          case 'F':
            report_temperature_flag_ = true;
            is_manager_key = true;
            break;
        }

        if (!is_manager_key && current_) {
          current_->PushStdinChar(ch);
        }
      }

      // Update gamepad data and buttons
      update_gamepad_data(gamepad_data_.RF_RX);

      // F1 - Toggle ZMQ streaming
      bool trigger_ZMQ_toggle = false;
      if (F1_.on_press) {
        if (active_ != ManagedType::ZMQ) {
          SetActiveInterface(ManagedType::ZMQ);
        }
        trigger_ZMQ_toggle = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] F1 pressed - ZMQ toggle" << std::endl;
        }
      }

      // D-pad - Motion set selection (switches back to GAMEPAD if in ZMQ)
      if (up_.on_press) {
        if (active_ != ManagedType::GAMEPAD) { SetActiveInterface(ManagedType::GAMEPAD); }
        selectMotionSet(0);  // Standing
      }
      if (down_.on_press) {
        if (active_ != ManagedType::GAMEPAD) { SetActiveInterface(ManagedType::GAMEPAD); }
        selectMotionSet(1);  // Squat/Crawl
      }
      if (left_.on_press) {
        if (active_ != ManagedType::GAMEPAD) { SetActiveInterface(ManagedType::GAMEPAD); }
        selectMotionSet(2);  // Boxing
      }
      if (right_.on_press) {
        if (active_ != ManagedType::GAMEPAD) { SetActiveInterface(ManagedType::GAMEPAD); }
        selectMotionSet(3);  // Styled Walking
      }

      // Select - Emergency Stop
      if (select_.on_press) {
        stop_control_ = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Select pressed - Emergency Stop" << std::endl;
        }
      }

      // If not in gamepad mode, update the active interface
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
                      std::mutex& current_motion_mutex,
                      bool& report_temperature) override {
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

      // Global temperature report (F key)
      if (report_temperature_flag_) {
        report_temperature = true;
        report_temperature_flag_ = false;
      }

      // Handle stop control
      if (stop_control_) { operator_state.stop = true; }

      // If in gamepad mode, handle planner-only controls
      if (active_ == ManagedType::GAMEPAD) {
        handleGamepadPlannerInput(motion_reader, current_motion, current_frame,
                                  operator_state, reinitialize_heading, heading_state_buffer,
                                  planner_state, movement_state_buffer, current_motion_mutex);
      } else {
        // Delegate to ZMQ
        if (current_) {
          current_->handle_input(motion_reader, current_motion, current_frame, operator_state,
                                reinitialize_heading, heading_state_buffer, has_planner,
                                planner_state, movement_state_buffer, current_motion_mutex, report_temperature);
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
    }

    void setActiveIndex(int idx) {
      if (order_.empty()) { return; }
      if (idx < 0) { idx = static_cast<int>(order_.size()) - 1; }
      if (idx >= static_cast<int>(order_.size())) { idx = 0; }

      // Trigger safety reset on all managed interfaces when switching
      TriggerSafetyReset();  // Self (for gamepad mode)
      if (zmq_) zmq_->TriggerSafetyReset();

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

    // Select a motion set by index and reset to its default mode
    void selectMotionSet(int set_index) {
      bool was_crawling = isInCrawlingMode();

      motion_set_index_ = set_index;
      current_motion_set_ = get_motion_set(motion_set_index_);
      mode_index_in_set_ = 0;
      boxing_revert_time_ = {};  // Clear boxing timer on set switch

      if (was_crawling) {
        // Soft exit: crawl → kneel → target (staged)
        LocomotionMode target = current_motion_set_[0];
        beginExitCrawlingTransition(target);
      } else {
        // Clear any pending transition and apply directly
        transition_target_mode_ = LocomotionMode::IDLE;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = {};
        applyModeFromSet();
      }

      // Boxing set enters with IDEL_BOXING → auto-revert to WALK_BOXING after 1s
      if (motion_set_index_ == 2) {
        boxing_revert_time_ = std::chrono::steady_clock::now();
      }
      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] D-pad - Motion set " << motion_set_index_
                  << ", mode: " << planner_use_movement_mode_ << std::endl;
      }
    }

    // Apply mode/speed/height from the current mode_index_in_set_
    void applyModeFromSet() {
      if (current_motion_set_.empty()) return;
      LocomotionMode mode = current_motion_set_[mode_index_in_set_];
      planner_use_movement_mode_ = static_cast<int>(mode);
      applySpeedAndHeight(mode);
    }

    // Set fixed speed and height for a given mode
    void applySpeedAndHeight(LocomotionMode mode) {
      switch (mode) {
        case LocomotionMode::SLOW_WALK:
          planner_use_movement_speed_ = 0.4;
          planner_use_height_ = -1.0;
          break;
        case LocomotionMode::RUN:
          planner_use_movement_speed_ = 1.5;
          planner_use_height_ = -1.0;
          break;
        case LocomotionMode::CRAWLING:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = 0.4;
          break;
        case LocomotionMode::ELBOW_CRAWLING:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = 0.3;
          break;
        case LocomotionMode::IDEL_SQUAT:
        case LocomotionMode::IDEL_KNEEL_TWO_LEGS:
        case LocomotionMode::IDEL_KNEEL:
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = 0.4;
          break;
        case LocomotionMode::IDEL_BOXING:
        case LocomotionMode::WALK_BOXING:
        case LocomotionMode::LEFT_PUNCH:
        case LocomotionMode::RIGHT_PUNCH:
        case LocomotionMode::RANDOM_PUNCH:
        case LocomotionMode::LEFT_HOOK:
        case LocomotionMode::RIGHT_HOOK:
          planner_use_movement_speed_ = 0.7;
          planner_use_height_ = -1.0;
          break;
        default:
          // WALK, styled walking, and other modes use defaults
          planner_use_movement_speed_ = -1.0;
          planner_use_height_ = -1.0;
          break;
      }
    }

    // Begin a staged crawling transition to target_mode, skipping completed stages
    void beginCrawlingTransition(LocomotionMode target_mode) {
      LocomotionMode current = static_cast<LocomotionMode>(planner_use_movement_mode_);

      if (target_mode == LocomotionMode::CRAWLING) {
        if (current == LocomotionMode::CRAWLING) return; // Already there
        if (current == LocomotionMode::IDEL_KNEEL_TWO_LEGS ||
            current == LocomotionMode::IDEL_KNEEL) {
          // Already kneeling, transition directly to crawling after delay
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else {
          // Need to kneel first
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        }
      } else if (target_mode == LocomotionMode::ELBOW_CRAWLING) {
        if (current == LocomotionMode::ELBOW_CRAWLING) return; // Already there
        if (current == LocomotionMode::CRAWLING) {
          // Already crawling, transition to elbow crawling after delay
          transition_target_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else if (current == LocomotionMode::IDEL_KNEEL_TWO_LEGS ||
                   current == LocomotionMode::IDEL_KNEEL) {
          // Already kneeling, go to crawling first then elbow
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_final_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        } else {
          // Need to kneel first, then crawl, then elbow
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
          transition_target_mode_ = LocomotionMode::CRAWLING;
          transition_final_mode_ = LocomotionMode::ELBOW_CRAWLING;
          transition_start_time_ = std::chrono::steady_clock::now();
        }
      }
      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Crawling transition started -> target: "
                  << static_cast<int>(target_mode) << std::endl;
      }
    }

    // Begin a staged exit from crawling/elbow crawling to target_mode (reverse of enter)
    // ELBOW_CRAWLING → (immediate CRAWLING) → (2s) KNEEL → (2s) target
    // CRAWLING → (immediate KNEEL) → (2s) target
    void beginExitCrawlingTransition(LocomotionMode target_mode) {
      LocomotionMode current = static_cast<LocomotionMode>(planner_use_movement_mode_);

      if (current == LocomotionMode::ELBOW_CRAWLING) {
        // Step down to CRAWLING immediately, then KNEEL after 2s, then target after 2s
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::CRAWLING);
        applySpeedAndHeight(LocomotionMode::CRAWLING);
        transition_target_mode_ = LocomotionMode::IDEL_KNEEL_TWO_LEGS;
        transition_final_mode_ = target_mode;
        transition_start_time_ = std::chrono::steady_clock::now();
      } else if (current == LocomotionMode::CRAWLING) {
        // Step down to KNEEL immediately, then target after 2s
        planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
        applySpeedAndHeight(LocomotionMode::IDEL_KNEEL_TWO_LEGS);
        transition_target_mode_ = target_mode;
        transition_start_time_ = std::chrono::steady_clock::now();
      }

      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Exit crawling transition -> target: "
                  << static_cast<int>(target_mode) << std::endl;
      }
    }

    // Check if current mode is crawling or elbow crawling
    bool isInCrawlingMode() const {
      return planner_use_movement_mode_ == static_cast<int>(LocomotionMode::CRAWLING) ||
             planner_use_movement_mode_ == static_cast<int>(LocomotionMode::ELBOW_CRAWLING);
    }

    // Process timed crawling transitions
    void processCrawlingTransitions() {
      if (transition_start_time_.time_since_epoch().count() == 0) return;

      auto elapsed = std::chrono::steady_clock::now() - transition_start_time_;
      if (elapsed < std::chrono::seconds(2)) return;

      // Time to advance to the next stage
      LocomotionMode next = transition_target_mode_;
      planner_use_movement_mode_ = static_cast<int>(next);
      applySpeedAndHeight(next);

      if constexpr (DEBUG_LOGGING) {
        std::cout << "[GamepadManager DEBUG] Transition -> " << static_cast<int>(next) << std::endl;
      }

      // Check if there's a further stage (elbow crawling after crawling)
      if (transition_final_mode_ != LocomotionMode::IDLE && next != transition_final_mode_) {
        transition_target_mode_ = transition_final_mode_;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = std::chrono::steady_clock::now();
      } else {
        // Done
        transition_target_mode_ = LocomotionMode::IDLE;
        transition_final_mode_ = LocomotionMode::IDLE;
        transition_start_time_ = {};
      }
    }

    // Process gamepad inputs for planner controls (called from update())
    void processGamepadPlannerControls() {
      // Start button
      if (start_.on_press) {
        start_control_ = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Start pressed" << std::endl;
        }
      }

      // A - Emergency Stop (always available)
      if (A_.on_press) {
        planner_emergency_stop_ = true;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] A pressed - Emergency Stop" << std::endl;
        }
      }

      // ---- Boxing set: direct key selection (no cycling) ----
      if (motion_set_index_ == 2) {
        if (X_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::WALK_BOXING);
          applySpeedAndHeight(LocomotionMode::WALK_BOXING);
          boxing_revert_time_ = {};  // WALK_BOXING is the default, no revert needed
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] X - WALK_BOXING" << std::endl; }
        }
        if (Y_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::RANDOM_PUNCH);
          applySpeedAndHeight(LocomotionMode::RANDOM_PUNCH);
          boxing_revert_time_ = {};  // RANDOM_PUNCH stays until user changes
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] Y - RANDOM_PUNCH" << std::endl; }
        }
        // B - IDEL_BOXING (auto-reverts to WALK_BOXING after 1s)
        if (B_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::IDEL_BOXING);
          applySpeedAndHeight(LocomotionMode::IDEL_BOXING);
          boxing_revert_time_ = std::chrono::steady_clock::now();
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] B - IDEL_BOXING (auto-revert 1s)" << std::endl; }
        }
        // L1/R1/L2/R2 - Punches/hooks (auto-revert to WALK_BOXING after 1s)
        if (L1_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::LEFT_PUNCH);
          applySpeedAndHeight(LocomotionMode::LEFT_PUNCH);
          boxing_revert_time_ = std::chrono::steady_clock::now();
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] L1 - LEFT_PUNCH (auto-revert 1s)" << std::endl; }
        }
        if (R1_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::RIGHT_PUNCH);
          applySpeedAndHeight(LocomotionMode::RIGHT_PUNCH);
          boxing_revert_time_ = std::chrono::steady_clock::now();
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] R1 - RIGHT_PUNCH (auto-revert 1s)" << std::endl; }
        }
        if (L2_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::LEFT_HOOK);
          applySpeedAndHeight(LocomotionMode::LEFT_HOOK);
          boxing_revert_time_ = std::chrono::steady_clock::now();
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] L2 - LEFT_HOOK (auto-revert 1s)" << std::endl; }
        }
        if (R2_.on_press) {
          planner_use_movement_mode_ = static_cast<int>(LocomotionMode::RIGHT_HOOK);
          applySpeedAndHeight(LocomotionMode::RIGHT_HOOK);
          boxing_revert_time_ = std::chrono::steady_clock::now();
          if constexpr (DEBUG_LOGGING) { std::cout << "[GamepadManager DEBUG] R2 - RIGHT_HOOK (auto-revert 1s)" << std::endl; }
        }

        // Auto-revert to WALK_BOXING after 1s if no new mode key pressed
        if (boxing_revert_time_.time_since_epoch().count() != 0) {
          auto elapsed = std::chrono::steady_clock::now() - boxing_revert_time_;
          if (elapsed > std::chrono::milliseconds(500)) {
            planner_use_movement_mode_ = static_cast<int>(LocomotionMode::WALK_BOXING);
            applySpeedAndHeight(LocomotionMode::WALK_BOXING);
            boxing_revert_time_ = {};
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] Boxing auto-revert -> WALK_BOXING" << std::endl;
            }
          }
        }
      } else {
        // ---- Standing / Squat / Styled sets: loop cycling ----
        if (X_.on_press) {
          // Next mode (wraps)
          bool was_crawling = isInCrawlingMode();
          mode_index_in_set_ = (mode_index_in_set_ + 1) % static_cast<int>(current_motion_set_.size());
          LocomotionMode target = current_motion_set_[mode_index_in_set_];
          if (target == LocomotionMode::CRAWLING || target == LocomotionMode::ELBOW_CRAWLING) {
            beginCrawlingTransition(target);
          } else if (was_crawling) {
            beginExitCrawlingTransition(target);
          } else {
            transition_start_time_ = {};  // Cancel any pending transition
            transition_target_mode_ = LocomotionMode::IDLE;
            transition_final_mode_ = LocomotionMode::IDLE;
            applyModeFromSet();
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] X - Next mode [" << mode_index_in_set_
                      << "]: " << planner_use_movement_mode_ << std::endl;
          }
        }
        if (Y_.on_press) {
          // Previous mode (wraps)
          bool was_crawling = isInCrawlingMode();
          mode_index_in_set_ = (mode_index_in_set_ - 1 + static_cast<int>(current_motion_set_.size()))
                               % static_cast<int>(current_motion_set_.size());
          LocomotionMode target = current_motion_set_[mode_index_in_set_];
          if (target == LocomotionMode::CRAWLING || target == LocomotionMode::ELBOW_CRAWLING) {
            beginCrawlingTransition(target);
          } else if (was_crawling) {
            beginExitCrawlingTransition(target);
          } else {
            transition_start_time_ = {};
            transition_target_mode_ = LocomotionMode::IDLE;
            transition_final_mode_ = LocomotionMode::IDLE;
            applyModeFromSet();
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Y - Prev mode [" << mode_index_in_set_
                      << "]: " << planner_use_movement_mode_ << std::endl;
          }
        }
        if (B_.on_press) {
          // Reset to set's default (index 0)
          bool was_crawling = isInCrawlingMode();
          mode_index_in_set_ = 0;
          LocomotionMode target = current_motion_set_[0];
          if (was_crawling && target != LocomotionMode::CRAWLING && target != LocomotionMode::ELBOW_CRAWLING) {
            beginExitCrawlingTransition(target);
          } else {
            transition_start_time_ = {};
            transition_target_mode_ = LocomotionMode::IDLE;
            transition_final_mode_ = LocomotionMode::IDLE;
            applyModeFromSet();
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] B - Reset to default mode: "
                      << planner_use_movement_mode_ << std::endl;
          }
        }

        // L1/R1 - Facing angle (not in boxing set)
        if (L1_.on_press) {
          planner_facing_angle_ += M_PI / 4;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] L1 - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
          }
        }
        if (R1_.on_press) {
          planner_facing_angle_ -= M_PI / 4;
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] R1 - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
          }
        }

        // L2/R2 - Height control (squat set only)
        if (motion_set_index_ == 1) {
          if (L2_.on_press) {
            planner_use_height_ -= 0.1;
            planner_use_height_ = std::max(planner_use_height_, 0.2);
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] L2 - Height: " << planner_use_height_ << std::endl;
            }
          }
          if (R2_.on_press) {
            planner_use_height_ += 0.1;
            planner_use_height_ = std::min(planner_use_height_, 0.8);
            if constexpr (DEBUG_LOGGING) {
              std::cout << "[GamepadManager DEBUG] R2 - Height: " << planner_use_height_ << std::endl;
            }
          }
        }
      }

      // Process timed crawling transitions (kneel → crawl → elbow)
      processCrawlingTransitions();

      // Analog sticks - facing and movement direction
      if (std::abs(rx_) > dead_zone_ || std::abs(ry_) > dead_zone_) {
        planner_facing_angle_ = planner_facing_angle_ - 0.02 * rx_;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Right stick - Facing angle: " << planner_facing_angle_ << " rad" << std::endl;
        }
      }

      if (std::abs(lx_) > dead_zone_ || std::abs(ly_) > dead_zone_) {
        double raw_angle = atan2(ly_, lx_);
        double bin_size = M_PI / 4.0;  // 8 directions (45° bins)
        double binned_angle = std::round(raw_angle / bin_size) * bin_size;
        planner_moving_direction_ = binned_angle - M_PI / 2 + planner_facing_angle_;
        if constexpr (DEBUG_LOGGING) {
          std::cout << "[GamepadManager DEBUG] Left stick - Raw: " << raw_angle
                    << ", Binned: " << binned_angle
                    << ", Moving: " << planner_moving_direction_ << " rad" << std::endl;
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

        // If left sticks in dead zone — idle logic depends on motion set
        if (std::abs(lx_) < dead_zone_ && std::abs(ly_) < dead_zone_) {
          if (motion_set_index_ == 0) {
            // Standing set → go to IDLE
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          } else if (motion_set_index_ == 1) {
            // Squat set → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            LocomotionMode cur = static_cast<LocomotionMode>(planner_use_movement_mode_);
            if (cur == LocomotionMode::CRAWLING || cur == LocomotionMode::ELBOW_CRAWLING) {
              final_speed = 0.0f;
            } else {
              final_speed = -1.0f;
            }
            final_height = planner_use_height_;
          } else if (motion_set_index_ == 2) {
            // Boxing set → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            LocomotionMode cur = static_cast<LocomotionMode>(planner_use_movement_mode_);
            if (cur == LocomotionMode::LEFT_PUNCH || cur == LocomotionMode::RIGHT_PUNCH ||
                cur == LocomotionMode::LEFT_HOOK || cur == LocomotionMode::RIGHT_HOOK) {
              // Punches use facing direction as movement
              final_movement = final_facing_direction;
            }
            final_speed = 0.0f;
            final_height = -1.0f;
          } else if (motion_set_index_ == 3) {
            // Styled walking → stay in mode, zero movement
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = -1.0f;
          }
        }

        // Emergency stop
        if (planner_emergency_stop_) {
          if (motion_set_index_ == 0) {
            final_mode = static_cast<int>(LocomotionMode::IDLE);
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = -1.0f;
            final_height = -1.0f;
          } else {
            final_movement = {0.0f, 0.0f, 0.0f};
            final_speed = 0.0f;
            final_height = (motion_set_index_ == 1) ? planner_use_height_ : -1.0f;
          }
          if constexpr (DEBUG_LOGGING) {
            std::cout << "[GamepadManager DEBUG] Emergency stop - movement reset" << std::endl;
          }
        }

        // Static kneel modes always zero movement
        if (planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL_TWO_LEGS) ||
            planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_KNEEL) ||
            planner_use_movement_mode_ == static_cast<int>(LocomotionMode::IDEL_SQUAT)) {
          final_movement = {0.0f, 0.0f, 0.0f};
          final_speed = 0.0f;
          final_height = planner_use_height_;
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
    /// Global temperature report flag, set by 'F'/'f' keyboard shortcut.
    bool report_temperature_flag_ = false;

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
    // Motion set state
    // ------------------------------------------------------------------
    int motion_set_index_ = 0;                    ///< Active motion set (0=Standing, 1=Squat, 2=Boxing, 3=Styled).
    std::vector<LocomotionMode> current_motion_set_;  ///< Modes in the active set.
    int mode_index_in_set_ = 0;                   ///< Current mode index within the set.

    // ------------------------------------------------------------------
    // Planner control state (persists across frames)
    // ------------------------------------------------------------------
    int planner_use_movement_mode_ = static_cast<int>(LocomotionMode::SLOW_WALK);  ///< Current locomotion mode.
    double planner_use_movement_speed_ = 0.4;    ///< Desired speed (fixed per mode).
    double planner_use_height_ = -1.0;            ///< Desired body height (−1 = mode default).
    double planner_facing_angle_ = 0.0;           ///< Accumulated facing direction (radians).
    double planner_moving_direction_ = 0.0;       ///< Current movement direction (radians).

    // ------------------------------------------------------------------
    // Staged crawling transition state
    // ------------------------------------------------------------------
    /// Target mode for the current transition stage (e.g., CRAWLING after kneeling).
    LocomotionMode transition_target_mode_ = LocomotionMode::IDLE;
    /// Final mode for multi-stage transitions (e.g., ELBOW_CRAWLING via kneel→crawl→elbow).
    LocomotionMode transition_final_mode_ = LocomotionMode::IDLE;
    /// Timestamp when the current transition stage began.
    std::chrono::time_point<std::chrono::steady_clock> transition_start_time_{};

    // ------------------------------------------------------------------
    // Boxing auto-revert state
    // ------------------------------------------------------------------
    /// Timestamp when a punch/hook/idle was triggered; reverts to WALK_BOXING after 1s.
    std::chrono::time_point<std::chrono::steady_clock> boxing_revert_time_{};
};

#endif // GAMEPAD_MANAGER_HPP


