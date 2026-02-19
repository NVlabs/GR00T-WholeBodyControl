/**
 * @file g1_deploy_onnx_ref.cpp
 * @brief Main application: deploys an RL locomotion policy on the Unitree G1 robot.
 *
 * This single-file application contains:
 *
 *   - **G1Deploy** class – the top-level controller that wires together all
 *     sub-systems (input, output, policy, encoder, planner, hands).
 *   - **main()** – CLI argument parser and run-loop.
 *
 * ## Threading Model
 *
 * Four real-time threads are created by the Unitree SDK's
 * `CreateRecurrentThreadEx`:
 *
 *   Thread           | Rate     | Method              | Responsibility
 *   -----------------|----------|---------------------|-------------------------------
 *   Input            | 100 Hz   | G1Deploy::Input     | Poll input interface, handle commands.
 *   Control          | 50 Hz    | G1Deploy::Control   | Gather obs, run policy, compute motor targets.
 *   Planner          | 10 Hz    | G1Deploy::Planner   | Re-plan locomotion trajectory.
 *   Command Writer   | 500 Hz   | G1Deploy::LowCommandWriter | Publish motor commands via DDS.
 *
 * ## Control-Loop State Machine (ProgramState)
 *
 *   INIT → WAIT_FOR_CONTROL → CONTROL
 *
 *   - **INIT**: Wait for a valid LowState message from the robot.
 *   - **WAIT_FOR_CONTROL**: Robot is ready; wait for operator "start" signal.
 *   - **CONTROL**: Active policy execution – gather observations, infer actions,
 *     write motor commands.  Exits on operator "stop" or error.
 *
 * ## CLI Arguments (selected)
 *
 *   Flag                  | Description
 *   ----------------------|------------
 *   --network             | DDS network interface (e.g. "eth0")
 *   --model               | Path to the policy ONNX model
 *   --motion-data         | Directory of pre-loaded reference motions
 *   --obs-config          | Observation config YAML
 *   --encoder-model       | Encoder ONNX model (for token_state)
 *   --planner-model       | Locomotion planner ONNX model
 *   --input-type          | keyboard / gamepad / zmq / ros2 / interface_manager / gamepad_manager / zmq_manager
 *   --output-type         | zmq / ros2 / all
 *   --disable-crc-check   | Skip CRC validation (for MuJoCo sim)
 *   --planner-fp16        | Use FP16 for planner TensorRT engine
 *   --policy-fp16         | Use FP16 for policy TensorRT engine
 */
#include <cmath>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <pthread.h>
#include <sched.h>
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// TRTInference
#include <TRTInference/InferenceEngine.h>

// ONNX
#include <onnxruntime_cxx_api.h>

// Motion Data Reader
#include "../include/motion_data_reader.hpp"

// Math utilities
#include "../include/math_utils.hpp"

// Observation configuration
#include "../include/observation_config.hpp"

// New Planner Classes
#include "../include/localmotion_kplanner.hpp"
#include "../include/localmotion_kplanner_tensorrt.hpp"

// Utility classes
#include "../include/utils.hpp"

// Robot parameters
#include "../include/robot_parameters.hpp"
#include "../include/policy_parameters.hpp"

// Input interface and input handlers
#include "../include/input_interface/keyboard_handler.hpp"
#include "../include/input_interface/gamepad.hpp"
#include "../include/input_interface/zmq_endpoint_interface.hpp"
#include "../include/input_interface/interface_manager.hpp"
#include "../include/input_interface/gamepad_manager.hpp"
#include "../include/input_interface/zmq_manager.hpp"

// Output interface and output handlers
#include "../include/output_interface/output_interface.hpp"

// Optional ROS2 input handler
#if HAS_ROS2
#include "../include/input_interface/ros2_input_handler.hpp"
#include "../include/output_interface/ros2_output_handler.hpp"
#endif

#include "../include/output_interface/zmq_output_handler.hpp"

#include <cuda_runtime.h>
#include "../include/state_logger.hpp"

// Encoder
#include "../include/encoder.hpp"

// Control policy
#include "../include/control_policy.hpp"

// Dex3 hands
#include "../include/dex3_hands.hpp"

#include "audio_thread/audio_thread.hpp"

// DDS
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;



/**
 * @class G1Deploy
 * @brief Top-level controller that orchestrates the entire G1 deployment pipeline.
 *
 * Owns and wires together:
 *  - InputInterface (keyboard / gamepad / ZMQ / ROS2 / managers)
 *  - PolicyEngine (TensorRT control policy)
 *  - EncoderEngine (optional TensorRT observation encoder)
 *  - LocalMotionPlannerBase (optional TensorRT locomotion planner)
 *  - Dex3Hands (optional Dex3 hand controller)
 *  - StateLogger (ring buffer + CSV persistence)
 *  - OutputInterface(s) (ZMQ / ROS2 state publishers)
 *  - MotionDataReader (pre-loaded reference motions)
 *
 * Four real-time threads handle Input (100 Hz), Control (50 Hz),
 * Planner (10 Hz), and Command Writing (500 Hz).
 */
class G1Deploy {
  private:
    /// State machine for the control loop lifecycle.
    enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL };
    
    // =========================================================================
    // Core timing, mode, and counters
    // =========================================================================
    double time_;          ///< Elapsed time since start (used during INIT ramp-up).
    double publish_dt_;    ///< Command writer period (500 Hz = 0.002 s).
    double control_dt_;    ///< Control loop period  (50 Hz = 0.02 s).
    double planner_dt_;    ///< Planner loop period  (10 Hz = 0.1 s).
    double input_dt_;      ///< Input poll period    (100 Hz = 0.01 s).
    double duration_;      ///< Duration of the INIT ramp-up to default pose (3 s).
    int counter_;          ///< General-purpose tick counter.
    Mode mode_pr_;         ///< Ankle control mode (series PR vs. parallel AB).
    uint8_t mode_machine_; ///< Robot variant code received from LowState.
    
    // =========================================================================
    // Input interface and buffered input data
    // =========================================================================
    std::unique_ptr<InputInterface> input_interface_;
    
    // Buffered input data (the real data to the policy engine)
    // has_*_data_ flags indicate whether the getter returned valid buffered data (true) or defaults (false)
    bool has_vr_3point_data_ = false;
    std::array<double, 9> vr_3point_position_buffer_;
    std::array<double, 12> vr_3point_orientation_buffer_;
    std::array<double, 3> vr_3point_compliance_buffer_;
    bool has_vr_5point_data_ = false;
    std::array<double, 15> vr_5point_position_buffer_;
    std::array<double, 20> vr_5point_orientation_buffer_;
    bool has_left_hand_data_ = false;
    bool has_right_hand_data_ = false;
    std::array<double, 7> left_hand_joint_buffer_;
    std::array<double, 7> right_hand_joint_buffer_;
    bool has_upper_body_data_ = false;
    std::array<double, 17> upper_body_joint_positions_buffer_;
    std::array<double, 17> upper_body_joint_velocities_buffer_;
    std::vector<double> token_state_data_;  // Token buffer (size from config)
    
    // =========================================================================
    // Motion data, current motion, and recording
    // =========================================================================
    // Motion data reader and current motion
    MotionDataReader motion_reader_;
    
    // Current motion and frame (using shared_ptr for thread safety)
    std::shared_ptr<const MotionSequence> current_motion_ = nullptr;
    int current_frame_ = 0;
    int saved_frame_for_observation_window_ = 0; // for observation window
    std::mutex current_motion_mutex_; // for current motion and frame synchronization
    
    // =========================================================================
    // Local motion planner and movement state
    // =========================================================================
    // New unified planner interface
    std::string planner_path;
    std::unique_ptr<LocalMotionPlannerBase> planner_;
    std::shared_ptr<MotionSequence> planner_motion_;
    
    // Movement momentum system
    MovementState last_movement_state_ = MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f);
    float replan_interval_running_ = 0.1;
    float replan_interval_crawling_ = 0.2;
    float replan_interval_boxing_ = 1.0;
    float replan_interval_ = 1.0;
    float replan_interval_counter_ = 0.0f;

    
    // =========================================================================
    // Low-level robot I/O buffers, channels, and threads
    // =========================================================================
    // Flag to disable CRC checking for MuJoCo simulation
    bool disable_crc_check_ = false;
    
    bool reinitialize_heading_ = false;
    std::array<double, 4> init_ref_data_root_rot_array_;

    DataBuffer<LowState_> low_state_buffer_;
    DataBuffer<MotorCommand> motor_command_buffer_;
    DataBuffer<IMUState_> imu_torso_buffer_;
    DataBuffer<HeadingState> heading_state_buffer_;
    DataBuffer<MovementState> movement_state_buffer_;
    
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;
    ThreadPtr input_thread_ptr_, command_writer_ptr_, control_thread_ptr_, planner_thread_ptr_;
    
    // =========================================================================
    // External clients and peripheral managers
    // =========================================================================
    std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;
    
    // Dex3 hands manager
    Dex3Hands dex3_hands_;

    static constexpr std::chrono::milliseconds STREAMING_DATA_ABSENT_THRESHOLD{150};
    CounterDebouncer streaming_data_absent_debouncer_{100, 500, 50, 1};
    RollingStats<1000> streaming_data_delay_rolling_stats_;
    std::unique_ptr<AudioThread> audio_thread_;
    
    // =========================================================================
    // Program state and last commanded actions
    // =========================================================================
    ProgramState program_state_;
    std::array<double, G1_NUM_MOTOR> last_action;
    std::array<double, 7> last_left_hand_action;
    std::array<double, 7> last_right_hand_action;
    
    // =========================================================================
    // Logging / recording streams
    // =========================================================================
    std::unique_ptr<std::ofstream> target_motion_file_;
    std::unique_ptr<std::ofstream> planner_motion_file_;
    std::unique_ptr<std::ofstream> policy_input_file_;
    std::unique_ptr<std::ofstream> record_input_file_;
    std::unique_ptr<std::ifstream> playback_input_file_;

    // Motion recorders for streamed and planner_motion
    MotionRecorder zmq_motion_recorder_;
    MotionRecorder planner_motion_recorder_;
    bool enable_motion_recording_ = false;
    
    // =========================================================================
    // Initial compliance values for VR 3-point control (set from command line)
    // These are keyboard-controlled and are the ONLY source of compliance values
    // =========================================================================
    std::array<double, 3> initial_vr_3point_compliance_ = {0.5, 0.5, 0.0};
    
    // Initial max close ratio for Dex3 hands (set from command line)
    // Default 1.0 allows full closure, use --max-close-ratio to limit
    // Keyboard controls (J/K) always available for runtime adjustment
    double initial_max_close_ratio_ = 1.0;
    
    // Track if vr_3point_compliance is observed by the policy
    // If false, adjusting compliance via keyboard has no effect on the policy
    bool has_vr_3point_compliance_obs_ = false;
    
    // Independent logging counter (not affected by planner buffer cleaning)
    int logging_counter_ = 0;
    TimestampedData<LowState_> used_low_state_data_;
    TimestampedData<IMUState_> used_imu_torso_data_;

    // State logger
    std::unique_ptr<StateLogger> state_logger_;
    
    // Output interfaces (supports multiple simultaneous outputs)
    std::vector<std::unique_ptr<OutputInterface>> output_interfaces_;

    // =========================================================================
    // Core ML components (encoder + policy) and interfaces
    // =========================================================================
    // Encoder engine
    std::string model_path;
    std::unique_ptr<EncoderEngine> encoder_engine_;
    EncoderConfig encoder_config_;  // Encoder configuration from observation config
    bool is_using_encoder_ = false;
    int initial_encoder_mode_ = -2;  // -2: no token state. -1: need token state but no encoder. 0,1,2,...: encoder mode.
    int last_logged_encoder_mode_ = -999;  // Track last logged mode to avoid spam
    
    // Control policy
    std::unique_ptr<PolicyEngine> policy_engine_;
    
    // =========================================================================
    // Observation system configuration and runtime state
    // =========================================================================
    // Function pointer type for observation functions (returns bool for success/failure)
    // Functions read from internal state (state_logger_, current_motion_, etc.)
    // and write to the provided target buffer at the given offset
    using ObservationFunction = std::function<bool(std::vector<double>&, size_t)>;
    
    // Observation registry - single place to define all observations
    struct ObservationRegistry {
      std::string name;
      size_t dimension;
      ObservationFunction function;
      
      ObservationRegistry(const std::string& n, size_t d, ObservationFunction f) 
        : name(n), dimension(d), function(f) {}
    };
 
    // Structure for active observation function with metadata
    struct ActiveObservation {
      std::string name;
      ObservationFunction function;
      size_t offset;
      size_t dimension;
      
      ActiveObservation(const std::string& n, ObservationFunction f, size_t off, size_t dim) 
        : name(n), function(f), offset(off), dimension(dim) {}
    };
    
    // Pre-allocated observation buffers to avoid allocation in control loop
    std::vector<double> obs_buffer_;          // Policy observation buffer
    std::vector<double> encoder_obs_buffer_;  // Encoder observation buffer
    
    // Observation configuration
    std::vector<ObservationConfig> obs_config_; 
    
    // Active observation functions (validated and ready for runtime)
    std::vector<ActiveObservation> active_obs_functions_;
    
    // Active encoder observation functions (for encoder input)
    std::vector<ActiveObservation> active_encoder_obs_functions_;
    
    // VR3Point index
    std::array<int, 3> actual_vr_3point_index = {-1, -1, -1};

    // VR5Point index
    std::array<int, 5> actual_vr_5point_index = {-1, -1, -1, -1, -1};

    // =========================================================================
    // Motion-based observation gatherers
    // 
    // These functions read from the currently-active MotionSequence to build
    // observation vectors for the policy neural network.  They support
    // multi-frame look-ahead (gathering future frames at a configurable
    // step interval) for temporal context.
    //
    // Each gatherer writes its output into `target_buffer` at `offset` and
    // updates `saved_frame_for_observation_window_` so that streamed motions
    // know how many look-ahead frames are needed before the end.
    // =========================================================================

    /// Gather root Z position from N future frames (1 value per frame).
    bool GatherMotionRootZPositionMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid body data
      if (current_motion_->GetNumBodies() == 0) {
        std::cerr << "✗ Error: Motion has no bodies - cannot gather root z position" << std::endl;
        return false;
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather root z position from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_root_z_pos = current_motion_->BodyPositions(target_frame)[0][2];
        target_buffer[offset + frame_idx * 1] = motion_root_z_pos;
      }
      return true;
    }

    /// Gather body positions from N future frames (3 values per body per frame).
    /// If body_part_indexes is empty, gathers all bodies; otherwise only the specified subset.
    bool GatherMotionBodyPositionsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> body_part_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid body data
      if (current_motion_->GetNumBodies() == 0) {
        std::cerr << "✗ Error: Motion has no bodies - cannot gather body positions" << std::endl;
        return false;
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather positions from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_body_pos = current_motion_->BodyPositions(target_frame);

        // If body part indexes are empty, gather all bodies (all x, y, z coordinates)
        if (body_part_indexes.empty()) {
          size_t num_bodies = current_motion_->GetNumBodies();
          size_t frame_offset = offset + frame_idx * num_bodies * 3;  // 3 coordinates per body
          
          std::vector<double> all_body_pos(num_bodies * 3);
          for (size_t i = 0; i < num_bodies; i++) {
            all_body_pos[i * 3 + 0] = motion_body_pos[i][0];  // x
            all_body_pos[i * 3 + 1] = motion_body_pos[i][1];  // y
            all_body_pos[i * 3 + 2] = motion_body_pos[i][2];  // z
          }
          std::copy(
            all_body_pos.begin(),
            all_body_pos.end(),
            target_buffer.begin() + frame_offset
          );
        // If body part indexes are not empty, gather only the needed bodies 
        } else {
          // Get motion's body part indexes to map desired body index to storage index
          const auto& motion_body_indexes = current_motion_->BodyPartIndexes();
          
          size_t frame_offset = offset + frame_idx * body_part_indexes.size() * 3;  // 3 coordinates per body
          std::vector<double> needed_body_pos(body_part_indexes.size() * 3);
          
          for (size_t i = 0; i < body_part_indexes.size(); i++) {
            int desired_body_index = body_part_indexes[i];
            
            // Find which storage index corresponds to this body index
            int storage_index = -1;
            for (size_t j = 0; j < motion_body_indexes.size(); j++) {
              if (motion_body_indexes[j] == desired_body_index) {
                storage_index = static_cast<int>(j);
                break;
              }
            }
            
            // Extract all x, y, z coordinates
            if (storage_index < 0) {
              std::cerr << "✗ Error: Body index " << desired_body_index 
                        << " not found in motion data. Available indices: [";
              for (size_t j = 0; j < motion_body_indexes.size(); j++) {
                std::cerr << motion_body_indexes[j];
                if (j + 1 < motion_body_indexes.size()) std::cerr << ", ";
              }
              std::cerr << "]" << std::endl;
              return false;
            }
            
            needed_body_pos[i * 3 + 0] = motion_body_pos[storage_index][0];  // x
            needed_body_pos[i * 3 + 1] = motion_body_pos[storage_index][1];  // y
            needed_body_pos[i * 3 + 2] = motion_body_pos[storage_index][2];  // z
          }
          std::copy(
            needed_body_pos.begin(),
            needed_body_pos.end(),
            target_buffer.begin() + frame_offset
          );
        }
      }
      return true;
    }

    /**
     * @brief Gather the heading-corrected anchor orientation from N future frames.
     *
     * Computes the relative rotation from the robot's current base orientation
     * to the heading-corrected reference orientation at each future frame.
     * Outputs the first two columns of the 3×3 rotation matrix (6 values per frame).
     *
     * Also handles heading reinitialisation when `reinitialize_heading_` is set.
     */
    bool GatherMotionAnchorOrientationMutiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      
      if (current_motion_->GetNumBodyQuaternions() == 0) {
        std::cout << "✗ Error: No motion_body_quat data available. Stopping control system." << std::endl;
        return false;
      }
      if (!state_logger_) { return false; }

      double sample_dt = control_dt_;
      auto hist = state_logger_->GetLatest(1, sample_dt);
      const auto& base_quat = hist[0].base_quat;


      if (reinitialize_heading_) {
        heading_state_buffer_.SetData(HeadingState(base_quat, 0.0));
        reinitialize_heading_ = false;
        const auto motion_body_quat_init = current_motion_->BodyQuaternions(current_frame_);
        init_ref_data_root_rot_array_ = motion_body_quat_init[0];
        std::cout << "Reset heading state to " << base_quat[0] << ", " << base_quat[1] << ", " << base_quat[2] << ", " << base_quat[3] << std::endl;
        std::cout << "Reset delta heading to 0" << std::endl;
        std::cout << "Reset init reference data root rotation to current frame: " << init_ref_data_root_rot_array_[0] << ", " << init_ref_data_root_rot_array_[1] << ", " << init_ref_data_root_rot_array_[2] << ", " << init_ref_data_root_rot_array_[3] << std::endl;
        std::cout << "Reference motion name: " << current_motion_->name << std::endl;
      }
      

      // Get heading state atomically (both quaternion and delta heading together)
      auto heading_state_data = heading_state_buffer_.GetDataWithTime().data;
      HeadingState heading_state = heading_state_data ? *heading_state_data : HeadingState();
      
      // Calculate initial heading from robot root pose (using double precision functions)
      auto init_heading = calc_heading_quat_d(heading_state.init_base_quat);
      
      // Calculate inverse heading from reference data
      if (current_frame_ == 0) { 
        const auto motion_body_quat_init = current_motion_->BodyQuaternions(0);
        init_ref_data_root_rot_array_ = motion_body_quat_init[0];
      }
      auto data_heading_inv = calc_heading_quat_inv_d(init_ref_data_root_rot_array_);
      
      // Apply delta heading calculation
      auto apply_delta_heading = quat_mul_d(init_heading, data_heading_inv);
      
      // Apply additional delta heading if specified
      if (heading_state.delta_heading != 0.0) {
        auto delta_quat = euler_z_to_quat_d(heading_state.delta_heading);
        apply_delta_heading = quat_mul_d(delta_quat, apply_delta_heading);
      }
      
      // Gather orientations from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }
        
        // Get reference data root rotation at this target frame (first body part, index 0)
        const auto motion_body_quat = current_motion_->BodyQuaternions(target_frame);
        std::array<double, 4> ref_data_root_rot_array = motion_body_quat[0];
        
        // Calculate new reference root rotation with heading applied
        auto new_ref_root_rot = quat_mul_d(apply_delta_heading, ref_data_root_rot_array);
        
        // Calculate motion_anchor_ori_b: relative rotation from base to reference at this timestep
        auto base_to_ref_quat = quat_mul_d(quat_conjugate_d(base_quat), new_ref_root_rot);
        auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);
        
        // Extract first 2 columns of rotation matrix and flatten ROW-WISE to 1D array (6 elements)
        // Python: .as_matrix()[..., :2].reshape(1, -1) flattens row by row
        std::array<double, 6> motion_anchor_ori_b = {
            rotation_matrix[0][0], rotation_matrix[0][1], // Row 0, cols 0-1
            rotation_matrix[1][0], rotation_matrix[1][1], // Row 1, cols 0-1
            rotation_matrix[2][0], rotation_matrix[2][1] // Row 2, cols 0-1
        };
        
        // Copy to observation buffer at the appropriate offset for this frame
        size_t frame_offset = offset + frame_idx * 6; // 6 values per frame
        std::copy(motion_anchor_ori_b.begin(), motion_anchor_ori_b.end(), target_buffer.begin() + frame_offset);
      }
      
      return true;
    }

    /// Gather joint positions from N future frames.  If joint_indexes is empty, gathers all 29.
    /// When upper-body control is active (has_upper_body_data_), replaces upper-body joints with
    /// the externally-provided targets.
    bool GatherMotionJointPositionsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> joint_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid joint data
      const auto num_joints = current_motion_->GetNumJoints();
      if (num_joints == 0) {
        std::cerr << "✗ Error: Motion has no joints - cannot gather joint positions" << std::endl;
        return false;
      }
      // Validate requested joint indexes
      for (int idx : joint_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_joints)) {
          std::cerr << "✗ Error: joint index " << idx
                    << " is out of range [0, " << (num_joints - 1)
                    << "] in GatherMotionJointPositionsMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather positions from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_joint_pos = current_motion_->JointPositions(target_frame);

        // If body part indexes are empty, gather all joints
        if (joint_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * 29;  // 29 joints per frame
          std::copy(
            motion_joint_pos,
            motion_joint_pos + num_joints,
            target_buffer.begin() + frame_offset
          );
          
          // If upper body control is enabled, use upper body joint positions in buffer to replace the motion_joint_pos
          if (has_upper_body_data_) {
            std::array<double, 29> current_motion_joint_pos;
            for (size_t i = 0; i < 29; i++) {
              current_motion_joint_pos[i] = motion_joint_pos[i];
            }
            for (size_t i = 0; i < 17; i++) {
              current_motion_joint_pos[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = upper_body_joint_positions_buffer_[i];
            }
            std::copy(
              current_motion_joint_pos.begin(),
              current_motion_joint_pos.end(),
              target_buffer.begin() + frame_offset
            );
          }

        // If body part indexes are not empty, gather only the needed joints
        } else {
          size_t frame_offset = offset + frame_idx * joint_indexes.size(); 
          std::vector<double> needed_joints_pos(joint_indexes.size());
          for (size_t i = 0; i < joint_indexes.size(); i++) {
            size_t body_part_index = joint_indexes[i];
            needed_joints_pos[i] = motion_joint_pos[body_part_index];
          }
          std::copy(
            needed_joints_pos.begin(),
            needed_joints_pos.end(),
            target_buffer.begin() + frame_offset
          );
        }
      }
      
      return true;
    }
    
    /// Gather joint velocities from N future frames.  Zeroes velocities when not playing.
    bool GatherMotionJointVelocitiesMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> joint_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid joint data
      const auto num_joints = current_motion_->GetNumJoints();
      if (num_joints == 0) {
        std::cerr << "✗ Error: Motion has no joints - cannot gather joint velocities" << std::endl;
        return false;
      }
      // Validate requested joint indexes
      for (int idx : joint_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_joints)) {
          std::cerr << "✗ Error: joint index " << idx
                    << " is out of range [0, " << (num_joints - 1)
                    << "] in GatherMotionJointVelocitiesMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather velocities from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate frame index (current frame + frame_idx * step_size for future frames)
        int target_frame = static_cast<int>(current_frame_) + frame_idx * step_size;
        
        // If beyond motion length, clamp to last frame (hold final pose)
        if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
          target_frame = static_cast<int>(current_motion_->timesteps) - 1;
        }
        
        const auto motion_joint_vel = current_motion_->JointVelocities(target_frame);

        // If body part indexes are empty, gather all joints
        if (joint_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * 29;  // 29 joints per frame
          if (operator_state.play) {
            std::copy(
              motion_joint_vel,
              motion_joint_vel + num_joints,
              target_buffer.begin() + frame_offset
            );
            // If upper body control is enabled, use upper body joint velocities in buffer to replace the motion_joint_vel
            if (has_upper_body_data_) {
              std::array<double, 29> current_motion_joint_vel;
              for (size_t i = 0; i < 29; i++) {
                current_motion_joint_vel[i] = motion_joint_vel[i];
              }
              for (size_t i = 0; i < 17; i++) {
                current_motion_joint_vel[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = upper_body_joint_velocities_buffer_[i];
              }
              std::copy(
                current_motion_joint_vel.begin(),
                current_motion_joint_vel.end(),
                target_buffer.begin() + frame_offset
              );
            }
          } else {
            std::fill_n(
              target_buffer.begin() + frame_offset,
              num_joints,
              0.0
            );
          }
        // If body part indexes are not empty, gather only the needed joints
        } else {
          size_t frame_offset = offset + frame_idx * joint_indexes.size(); 
          if (operator_state.play) {
            std::vector<double> needed_joints_vel(joint_indexes.size());
            for (size_t i = 0; i < joint_indexes.size(); i++) {
              size_t body_part_index = joint_indexes[i];
              needed_joints_vel[i] = motion_joint_vel[body_part_index];
            }
            std::copy(
              needed_joints_vel.begin(),
              needed_joints_vel.end(),
              target_buffer.begin() + frame_offset
            );
          } else {
            std::fill_n(
              target_buffer.begin() + frame_offset,
              joint_indexes.size(),
              0.0
            );
          }
        }
      }
      
      return true;
    }
    
    /// Gather SMPL joint positions from N future frames (3 values × num_smpl_joints per frame).
    bool GatherMotionSmplJointsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> smpl_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid SMPL joint data
      const auto num_smpl_joints = current_motion_->GetNumSmplJoints();
      if (num_smpl_joints == 0) {
        std::cerr << "✗ Error: Motion has no SMPL joints - cannot gather SMPL joint data" << std::endl;
        return false;
      }
      // Validate requested SMPL joint indexes
      for (int idx : smpl_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_smpl_joints)) {
          std::cerr << "✗ Error: SMPL joint index " << idx
                    << " is out of range [0, " << (num_smpl_joints - 1)
                    << "] in GatherMotionSmplJointsMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      
      // Gather SMPL joints from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }
        
        const auto motion_smpl_joints = current_motion_->SmplJoints(target_frame);

        // If indexes are empty, gather all SMPL joints (all x, y, z coordinates)
        if (smpl_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * num_smpl_joints * 3;  // 3 coordinates per joint
          
          for (size_t j = 0; j < num_smpl_joints; ++j) {
            target_buffer[frame_offset + j * 3 + 0] = motion_smpl_joints[j][0]; // x
            target_buffer[frame_offset + j * 3 + 1] = motion_smpl_joints[j][1]; // y
            target_buffer[frame_offset + j * 3 + 2] = motion_smpl_joints[j][2]; // z
          }
        // If indexes are not empty, gather only the needed SMPL joints
        } else {
          size_t frame_offset = offset + frame_idx * smpl_indexes.size() * 3; 
          for (size_t i = 0; i < smpl_indexes.size(); i++) {
            size_t smpl_idx = smpl_indexes[i];
            target_buffer[frame_offset + i * 3 + 0] = motion_smpl_joints[smpl_idx][0]; // x
            target_buffer[frame_offset + i * 3 + 1] = motion_smpl_joints[smpl_idx][1]; // y
            target_buffer[frame_offset + i * 3 + 2] = motion_smpl_joints[smpl_idx][2]; // z
          }
        }
      }
      
      return true;
    }
    
    /// Gather SMPL pose (axis-angle) data from N future frames (3 values × num_smpl_poses per frame).
    bool GatherMotionSmplPosesMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> pose_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid SMPL pose data
      const auto num_smpl_poses = current_motion_->GetNumSmplPoses();
      if (num_smpl_poses == 0) {
        std::cerr << "✗ Error: Motion has no SMPL poses - cannot gather SMPL pose data" << std::endl;
        return false;
      }
      // Validate requested SMPL pose indexes
      for (int idx : pose_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_smpl_poses)) {
          std::cerr << "✗ Error: SMPL pose index " << idx
                    << " is out of range [0, " << (num_smpl_poses - 1)
                    << "] in GatherMotionSmplPosesMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      
      // Gather SMPL poses from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }
        
        const auto motion_smpl_poses = current_motion_->SmplPoses(target_frame);

        // If indexes are empty, gather all SMPL poses (all x, y, z coordinates)
        if (pose_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * num_smpl_poses * 3;  // 3 coordinates per pose
          
          for (size_t p = 0; p < num_smpl_poses; ++p) {
            target_buffer[frame_offset + p * 3 + 0] = motion_smpl_poses[p][0]; // x
            target_buffer[frame_offset + p * 3 + 1] = motion_smpl_poses[p][1]; // y
            target_buffer[frame_offset + p * 3 + 2] = motion_smpl_poses[p][2]; // z
          }
        // If indexes are not empty, gather only the needed SMPL poses
        } else {
          size_t frame_offset = offset + frame_idx * pose_indexes.size() * 3; 
          for (size_t i = 0; i < pose_indexes.size(); i++) {
            size_t pose_idx = pose_indexes[i];
            target_buffer[frame_offset + i * 3 + 0] = motion_smpl_poses[pose_idx][0]; // x
            target_buffer[frame_offset + i * 3 + 1] = motion_smpl_poses[pose_idx][1]; // y
            target_buffer[frame_offset + i * 3 + 2] = motion_smpl_poses[pose_idx][2]; // z
          }
        }
      }
      
      return true;
    }
    
    // =========================================================================
    // VR 3/5-point observation helpers and gatherers
    //
    // VR tracking data can come from two sources:
    //   1. External input interface (ZMQ / ROS2) → use buffered values directly.
    //   2. Computed from motion body positions with per-body offsets, then
    //      normalised to the root body frame.
    //
    // The UpdateVR*PointIndices() helpers map the canonical VR body indices
    // (e.g. left wrist = body 28) to storage indices in the current
    // motion's BodyPartIndexes array (which varies between motions).
    // =========================================================================

    /// Map canonical VR 3-point body indices to current motion's storage indices.
    /// @return True if all 3 required indices were found.
    bool UpdateVR3PointIndices() {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }
      
      auto body_part_indexes = current_motion_->BodyPartIndexes();
      
      // Reset indices
      actual_vr_3point_index = {-1, -1, -1};
      
      // Find positions of required indices
      for (size_t i = 0; i < body_part_indexes.size(); i++) {
        for (size_t j = 0; j < vr_3point_index.size(); j++) {
          if (body_part_indexes[i] == vr_3point_index[j] && actual_vr_3point_index[j] == -1) {
            actual_vr_3point_index[j] = static_cast<int>(i);
            break; // Take first occurrence, move to next body part
          }
        }
      }
      
      // Validate that all required indices were found
      for (size_t j = 0; j < actual_vr_3point_index.size(); j++) {
        if (actual_vr_3point_index[j] == -1) {
          // VR3Point index not found - will fall back to interface values
          return false;
        }
      }
      
      return true;
    }

    // Helper function to update VR3Point indices for current motion
    // Returns true if all indices found, false otherwise
    bool UpdateVR5PointIndices() {
      if (!current_motion_ || current_motion_->timesteps == 0) { return false; }

      auto body_part_indexes = current_motion_->BodyPartIndexes();

      // Reset indices
      actual_vr_5point_index = {-1, -1, -1, -1, -1};

      // Find positions of required indices
      for (size_t i = 0; i < body_part_indexes.size(); i++) {
        for (size_t j = 0; j < vr_5point_index.size(); j++) {
          if (body_part_indexes[i] == vr_5point_index[j] && actual_vr_5point_index[j] == -1) {
            actual_vr_5point_index[j] = static_cast<int>(i);
            break; // Take first occurrence, move to next body part
          }
        }
      }

      // Validate that all required indices were found
      for (size_t j = 0; j < actual_vr_5point_index.size(); j++) {
        if (actual_vr_5point_index[j] == -1) {
          // VR5Point index not found - will fall back to interface values
          return false;
        }
      }

      return true;
    }

    bool GatherVR3PointPosition(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }


      // 1. Input interface providing VR 3-point (ZMQ, ROS2) -> use buffer
      // 2. Otherwise -> compute from motion body positions with offsets
      bool use_buffered_vr3 = has_vr_3point_data_;
      
      if (use_buffered_vr3) {
        // Use VR 3-point data from input interface directly
        // The sender (ZMQ/ROS2/planner) has already applied offsets and root normalization
        std::copy(vr_3point_position_buffer_.begin(), vr_3point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // No external VR 3-point data - compute from motion's body positions
      // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
      // If indices not found, fall back to buffered interface values (fallback to default values)
      if (!UpdateVR3PointIndices()) {
        std::copy(vr_3point_position_buffer_.begin(), vr_3point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // read from current motion
      int frame = current_frame_ % current_motion_->timesteps;
      const auto& motion_body_pos = current_motion_->BodyPositions(frame);
      const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

      // VR 3-point offsets in body's local frame (matching ros2_input_handler.hpp and visualize_motion.py)
      // Order: left_wrist (index 0), right_wrist (index 1), head/torso (index 2)
      // NOTE: Y-axis is mirrored for right wrist (symmetric robot)
      constexpr std::array<std::array<double, 3>, 3> VR_3POINT_OFFSETS = {{
        {0.18, -0.025, 0.0},   // left wrist offset
        {0.18, +0.025, 0.0},   // right wrist offset (Y mirrored for symmetry)
        {0.0, 0.0, 0.35}       // head offset (applied to torso body)
      }};

      // Apply offsets to body positions to get VR 3-point positions
      std::array<double, 9> vr_3point_position;
      for (int i = 0; i < 3; ++i) {
        // Get body position and orientation
        const auto& body_pos = motion_body_pos[actual_vr_3point_index[i]];
        const auto& body_quat = motion_body_quat[actual_vr_3point_index[i]];
        
        // Rotate offset by body orientation to get offset in world frame
        auto offset_world = quat_rotate_d(body_quat, VR_3POINT_OFFSETS[i]);
        
        // Add rotated offset to body position to get VR point position
        vr_3point_position[i * 3 + 0] = body_pos[0] + offset_world[0];
        vr_3point_position[i * 3 + 1] = body_pos[1] + offset_world[1];
        vr_3point_position[i * 3 + 2] = body_pos[2] + offset_world[2];
      }

      // normalize positions to the root by - root_pos rotate and normalize the root_quat
      auto root_pos = current_motion_->BodyPositions(frame)[0];
      auto root_quat = current_motion_->BodyQuaternions(frame)[0];
      
      // Normalize each VR 3-point position relative to root
      // Apply inverse transformation: subtract root position, then rotate by inverse root quaternion
      auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
      
      for (int i = 0; i < 3; ++i) {
        // Extract position for this VR point
        std::array<double, 3> point_pos = {
          vr_3point_position[i * 3 + 0],
          vr_3point_position[i * 3 + 1], 
          vr_3point_position[i * 3 + 2]
        };
        
        // Make position relative to root (subtract root position)
        std::array<double, 3> relative_pos = {
          point_pos[0] - root_pos[0],
          point_pos[1] - root_pos[1],
          point_pos[2] - root_pos[2]
        };
        
        // Rotate relative position by inverse root quaternion
        auto normalized_pos = quat_rotate_d(root_quat_inv, relative_pos);
        
        // Store normalized position back to the array
        vr_3point_position[i * 3 + 0] = normalized_pos[0];
        vr_3point_position[i * 3 + 1] = normalized_pos[1];
        vr_3point_position[i * 3 + 2] = normalized_pos[2];
      }
      
      std::copy(
        vr_3point_position.begin(),
        vr_3point_position.end(),
        target_buffer.begin() + offset
      );
      
      // Update buffer with the computed value from dataset 
      vr_3point_position_buffer_ = vr_3point_position;
      
      return true;
    }

    bool GatherVR3PointOrientation(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      
      bool use_buffered_vr3 = has_vr_3point_data_;

      if (use_buffered_vr3) {
        std::copy(vr_3point_orientation_buffer_.begin(), vr_3point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // No external VR 3-point data - compute from motion's body positions
      // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
      // If indices not found, fall back to buffered interface values (fallback to default values)
      if (!UpdateVR3PointIndices()) {
        std::copy(vr_3point_orientation_buffer_.begin(), vr_3point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }

      // read from current motion
      int frame = current_frame_ % current_motion_->timesteps;
      const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

      // Copy quaternions to observation buffer (flatten 3x4D quaternions to 12 values)
      std::array<double, 12> vr_3point_orientation = {
        motion_body_quat[actual_vr_3point_index[0]][0], motion_body_quat[actual_vr_3point_index[0]][1], motion_body_quat[actual_vr_3point_index[0]][2], motion_body_quat[actual_vr_3point_index[0]][3], // Quat 0: w,x,y,z
        motion_body_quat[actual_vr_3point_index[1]][0], motion_body_quat[actual_vr_3point_index[1]][1], motion_body_quat[actual_vr_3point_index[1]][2], motion_body_quat[actual_vr_3point_index[1]][3], // Quat 1: w,x,y,z
        motion_body_quat[actual_vr_3point_index[2]][0], motion_body_quat[actual_vr_3point_index[2]][1], motion_body_quat[actual_vr_3point_index[2]][2], motion_body_quat[actual_vr_3point_index[2]][3]  // Quat 2: w,x,y,z
      };

      // Normalize quaternions relative to root quaternion: quat_mul(quat_inv(root_quat), vr_quat)
      auto root_quat = current_motion_->BodyQuaternions(frame)[0];
      auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
      
      for (int i = 0; i < 3; ++i) {
        // Extract quaternion for this VR point
        std::array<double, 4> point_quat = {
          vr_3point_orientation[i * 4 + 0], // w
          vr_3point_orientation[i * 4 + 1], // x
          vr_3point_orientation[i * 4 + 2], // y
          vr_3point_orientation[i * 4 + 3]  // z
        };
        
        // Apply quaternion normalization: quat_mul(quat_inv(root_quat), vr_quat)
        auto normalized_quat = quat_mul_d(root_quat_inv, point_quat);
        
        // Store normalized quaternion back to the array
        vr_3point_orientation[i * 4 + 0] = normalized_quat[0]; // w
        vr_3point_orientation[i * 4 + 1] = normalized_quat[1]; // x
        vr_3point_orientation[i * 4 + 2] = normalized_quat[2]; // y
        vr_3point_orientation[i * 4 + 3] = normalized_quat[3]; // z
      }
      
      std::copy(
        vr_3point_orientation.begin(),
        vr_3point_orientation.end(),
        target_buffer.begin() + offset
      );
      
      // Update buffer with the computed value from dataset
      vr_3point_orientation_buffer_ = vr_3point_orientation;
      
      return true;
    }


    bool GatherVR5PointPosition(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }

      if (current_motion_ && current_motion_ == planner_motion_){
        // if planner motion, read from buffered interface data
        std::copy(vr_5point_position_buffer_.begin(), vr_5point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      } else {
        // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
        // If indices not found, fall back to buffered interface values
        if (!UpdateVR5PointIndices()) {
          std::copy(vr_5point_position_buffer_.begin(), vr_5point_position_buffer_.end(), target_buffer.begin() + offset);
          return true;
        }
        
        // read from current motion
        int frame = current_frame_ % current_motion_->timesteps;
        const auto& motion_body_pos = current_motion_->BodyPositions(frame);
        const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

        // VR 3-point offsets in body's local frame (matching ros2_input_handler.hpp and visualize_motion.py)
        // Order: left_wrist (index 0), right_wrist (index 1), head/torso (index 2)
        // NOTE: Y-axis is mirrored for right wrist (symmetric robot)
        constexpr std::array<std::array<double, 3>, 5> VR_5POINT_OFFSETS = {
          std::array<double, 3>{0.0, 0.0, 0.0},   // left wrist offset
          std::array<double, 3>{0.0, 0.0, 0.0},   // right wrist offset (Y mirrored for symmetry)
          std::array<double, 3>{0.0, 0.0, 0.0},   // head offset (applied to torso body)
          std::array<double, 3>{0.0, 0.0, 0.0},   // left ankle offset
          std::array<double, 3>{0.0, 0.0, 0.0}    // right ankle offset
        };

        // Apply offsets to body positions to get VR 3-point positions
        std::array<double, 15> vr_5point_position;
        for (int i = 0; i < 5; ++i) {
          // Get body position and orientation
          const auto& body_pos = motion_body_pos[actual_vr_5point_index[i]];
          const auto& body_quat = motion_body_quat[actual_vr_5point_index[i]];
          
          // Rotate offset by body orientation to get offset in world frame
          auto offset_world = quat_rotate_d(body_quat, VR_5POINT_OFFSETS[i]);
          
          // Add rotated offset to body position to get VR point position
          vr_5point_position[i * 3 + 0] = body_pos[0] + offset_world[0];
          vr_5point_position[i * 3 + 1] = body_pos[1] + offset_world[1];
          vr_5point_position[i * 3 + 2] = body_pos[2] + offset_world[2];
        }

        // normalize positions to the root by - root_pos rotate and normalize the root_quat
        auto root_pos = current_motion_->BodyPositions(frame)[0];
        auto root_quat = current_motion_->BodyQuaternions(frame)[0];
        
        // Normalize each VR 3-point position relative to root
        // Apply inverse transformation: subtract root position, then rotate by inverse root quaternion
        auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
        
        for (int i = 0; i < 5; ++i) {
          // Extract position for this VR point
          std::array<double, 3> point_pos = {
            vr_5point_position[i * 3 + 0],
            vr_5point_position[i * 3 + 1], 
            vr_5point_position[i * 3 + 2]
          };
          
          // Make position relative to root (subtract root position)
          std::array<double, 3> relative_pos = {
            point_pos[0] - root_pos[0],
            point_pos[1] - root_pos[1],
            point_pos[2] - root_pos[2]
          };
          
          // Rotate relative position by inverse root quaternion
          auto normalized_pos = quat_rotate_d(root_quat_inv, relative_pos);
          
          // Store normalized position back to the array
          vr_5point_position[i * 3 + 0] = normalized_pos[0];
          vr_5point_position[i * 3 + 1] = normalized_pos[1];
          vr_5point_position[i * 3 + 2] = normalized_pos[2];
        }
        
        std::copy(
          vr_5point_position.begin(),
          vr_5point_position.end(),
          target_buffer.begin() + offset
        );
        
        // Update buffer with the computed value from dataset 
        vr_5point_position_buffer_ = vr_5point_position;
        
        return true;
      }
      return false;
    }

    bool GatherVR5PointOrientation(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      
      if (current_motion_ && current_motion_ == planner_motion_){
        std::copy(vr_5point_orientation_buffer_.begin(), vr_5point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      } else {
        // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
        // If indices not found, fall back to buffered interface values
        if (!UpdateVR5PointIndices()) {
          std::copy(vr_5point_orientation_buffer_.begin(), vr_5point_orientation_buffer_.end(), target_buffer.begin() + offset);
          return true;
        }
        
        // read from current motion
        int frame = current_frame_ % current_motion_->timesteps;
        const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

        // Copy quaternions to observation buffer (flatten 3x4D quaternions to 12 values)
        std::array<double, 20> vr_5point_orientation = {
          motion_body_quat[actual_vr_5point_index[0]][0], motion_body_quat[actual_vr_5point_index[0]][1], motion_body_quat[actual_vr_5point_index[0]][2], motion_body_quat[actual_vr_5point_index[0]][3], // Quat 0: w,x,y,z
          motion_body_quat[actual_vr_5point_index[1]][0], motion_body_quat[actual_vr_5point_index[1]][1], motion_body_quat[actual_vr_5point_index[1]][2], motion_body_quat[actual_vr_5point_index[1]][3], // Quat 1: w,x,y,z
          motion_body_quat[actual_vr_5point_index[2]][0], motion_body_quat[actual_vr_5point_index[2]][1], motion_body_quat[actual_vr_5point_index[2]][2], motion_body_quat[actual_vr_5point_index[2]][3], // Quat 2: w,x,y,z
          motion_body_quat[actual_vr_5point_index[3]][0], motion_body_quat[actual_vr_5point_index[3]][1], motion_body_quat[actual_vr_5point_index[3]][2], motion_body_quat[actual_vr_5point_index[3]][3], // Quat 3: w,x,y,z
          motion_body_quat[actual_vr_5point_index[4]][0], motion_body_quat[actual_vr_5point_index[4]][1], motion_body_quat[actual_vr_5point_index[4]][2], motion_body_quat[actual_vr_5point_index[4]][3]  // Quat 4: w,x,y,z
        };

        // Normalize quaternions relative to root quaternion: quat_mul(quat_inv(root_quat), vr_quat)
        auto root_quat = current_motion_->BodyQuaternions(frame)[0];
        auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
        
        for (int i = 0; i < 5; ++i) {
          // Extract quaternion for this VR point
          std::array<double, 4> point_quat = {
            vr_5point_orientation[i * 4 + 0], // w
            vr_5point_orientation[i * 4 + 1], // x
            vr_5point_orientation[i * 4 + 2], // y
            vr_5point_orientation[i * 4 + 3]  // z
          };
          
          // Apply quaternion normalization: quat_mul(quat_inv(root_quat), vr_quat)
          auto normalized_quat = quat_mul_d(root_quat_inv, point_quat);
          
          // Store normalized quaternion back to the array
          vr_5point_orientation[i * 4 + 0] = normalized_quat[0]; // w
          vr_5point_orientation[i * 4 + 1] = normalized_quat[1]; // x
          vr_5point_orientation[i * 4 + 2] = normalized_quat[2]; // y
          vr_5point_orientation[i * 4 + 3] = normalized_quat[3]; // z
        }
        
        std::copy(
          vr_5point_orientation.begin(),
          vr_5point_orientation.end(),
          target_buffer.begin() + offset
        );
        
        // Update buffer with the computed value from dataset
        vr_5point_orientation_buffer_ = vr_5point_orientation;
        
        return true;
      }
      return false;
    }

    bool GatherVR3PointCompliance(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Compliance always comes from interface (not from motion dataset)
      std::copy(vr_3point_compliance_buffer_.begin(), vr_3point_compliance_buffer_.end(), target_buffer.begin() + offset);
      return true;
    }

    // =========================================================================
    // History-based observation gatherers (from StateLogger)
    //
    // These functions read past robot state from the StateLogger ring buffer
    // to provide temporal context to the policy.  They support multi-frame
    // history with configurable step size (e.g., 4 frames at 1× dt = last
    // 4 ticks; 10 frames at 2× dt = last 20 ticks sampled every other one).
    //
    // Each returns false if the required historical data is unavailable.
    // =========================================================================

    /// Gather historical body joint positions (num_frames × joints_per_frame).
    bool GatherHisBodyJointPositions(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      // step_size here means: sample every step_size * dt (e.g., step_size=2 means 0, 2*dt, 4*dt, ...)
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].body_q.empty()) {
        std::cerr << "✗ Error: No historical joint position data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].body_q.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisBodyJointPositions" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          // All joints
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.body_q.empty()) {
            std::copy(entry.body_q.begin(), entry.body_q.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          // Specific joints
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.body_q.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.body_q.size()) ? entry.body_q[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisBodyJointVelocities(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].body_dq.empty()) {
        std::cerr << "✗ Error: No historical joint velocity data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].body_dq.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisBodyJointVelocities" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.body_dq.empty()) {
            std::copy(entry.body_dq.begin(), entry.body_dq.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.body_dq.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.body_dq.size()) ? entry.body_dq[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisLastActions(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].last_action.empty()) {
        std::cerr << "✗ Error: No historical last action data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].last_action.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisLastActions" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.last_action.empty()) {
            std::copy(entry.last_action.begin(), entry.last_action.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.last_action.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.last_action.size()) ? entry.last_action[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisBaseAngularVelocity(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty()) {
        std::cerr << "✗ Error: No historical base angular velocity data available" << std::endl;
        return false;
      }
      
      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        size_t frame_offset = offset + static_cast<size_t>(f) * 3;
        target_buffer[frame_offset + 0] = entry.base_ang_vel[0];
        target_buffer[frame_offset + 1] = entry.base_ang_vel[1];
        target_buffer[frame_offset + 2] = entry.base_ang_vel[2];
      }
      return true;
    }

    bool GatherHisGravityDir(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty()) {
        std::cerr << "✗ Error: No historical gravity direction data available" << std::endl;
        return false;
      }
      
      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        auto gravity_dir_inv = quat_conjugate_d(entry.base_quat);
        std::array<double, 3> gravity_dir = quat_rotate_d(gravity_dir_inv, {0.0, 0.0, -1.0});
        size_t frame_offset = offset + static_cast<size_t>(f) * 3;
        target_buffer[frame_offset + 0] = gravity_dir[0];
        target_buffer[frame_offset + 1] = gravity_dir[1];
        target_buffer[frame_offset + 2] = gravity_dir[2];
      }
      return true;
    }

    // =========================================================================
    // Token / encoder observation gatherers
    //
    // The token state can be produced in two ways:
    //   1. Local encoder: GatherEncoderObservations() fills the encoder's
    //      input buffer, Encode() runs TensorRT inference, and the output
    //      tokens are copied into token_state_data_.
    //   2. External source: token_state_data_ was already populated in
    //      GatherInputInterfaceData() from ZMQ / ROS2.
    // =========================================================================

    /// Populate the token_state observation (either from local encoder or external data).
    bool GatherTokenState(std::vector<double>& target_buffer, size_t offset) {
      if (!is_using_encoder_) {
        // No encoder configured; use token_state_data_ (can be set externally via ROS2/ZMQ)
        std::copy(token_state_data_.begin(), token_state_data_.end(), target_buffer.begin() + offset);
        return true;
      }

      if (!encoder_engine_) {
        std::cerr << "✗ Error: Encoder engine not initialized!" << std::endl;
        return false;
      }

      // Gather encoder observations (populate encoder's internal input buffer)
      if (!GatherEncoderObservations()) {
        std::cerr << "✗ Error: Failed to gather encoder observations!" << std::endl;
        return false;
      }

      // Run encoder inference (handles CPU→GPU transfer, inference, GPU→CPU transfer)
      if (!encoder_engine_->Encode()) {
        std::cerr << "✗ Error: Encoder inference failed" << std::endl;
        return false;
      }

      // Access encoded tokens from encoder's internal buffer (already populated by Encode)
      auto& token_buffer = encoder_engine_->GetTokenBuffer();
      const size_t token_dim = encoder_engine_->GetTokenDimension();
      
      // Validate token dimension matches expected size
      if (token_dim != token_state_data_.size()) {
        std::cerr << "✗ Error: Encoder output dimension (" << token_dim 
                  << ") doesn't match token state buffer size (" << token_state_data_.size() << ")!" << std::endl;
        return false;
      }
      
      for (size_t i = 0; i < token_dim; ++i) {
        const double v = static_cast<double>(token_buffer[i]);
        token_state_data_[i] = v;
        target_buffer[offset + i] = v;
      }
      return true;
    }

    bool GatherEncoderMode(std::vector<double>& target_buffer, size_t offset, int fill_zeros_num = 0) {
      target_buffer[offset] = static_cast<float>(current_motion_->GetEncodeMode());
      for (int i = 1; i <= fill_zeros_num; ++i) {
        target_buffer[offset + i] = 0;
      }
      return true;
    }

    // =========================================================================
    // Observation registry
    //
    // Single source of truth mapping observation names (as they appear in
    // observation_config.yaml) to their dimension and gatherer lambda.
    // Both the policy and encoder observation systems use this registry.
    // =========================================================================

    /// Build the complete observation registry (name, dimension, gatherer function).
    std::vector<ObservationRegistry> GetObservationRegistry() {
      // Use token dimension from encoder config (defaults to 64 if not configured)
      size_t token_dim = (encoder_config_.dimension > 0) ? encoder_config_.dimension : 64;
      return {{"token_state", token_dim, [this](std::vector<double>& buf, size_t offset) { return GatherTokenState(buf, offset); }},
              {"encoder_mode", 3, [this](std::vector<double>& buf, size_t offset) { return GatherEncoderMode(buf, offset, 2); }},
              {"encoder_mode_4", 4, [this](std::vector<double>& buf, size_t offset) { return GatherEncoderMode(buf, offset, 3); }},
              {"motion_joint_positions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 1, 1); }},
              {"motion_joint_velocities", 29, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 1, 1); }},
              {"motion_anchor_orientation", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1); }},
              {"motion_root_z_position", 1, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 1, 1); }},
              {"motion_root_z_position_10frame_step5", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 5); }},
              {"motion_root_z_position_10frame_step1", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 1); }},
              {"motion_root_z_position_3frame_step1", 3, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 3, 1); }},
              {"motion_anchor_orientation_10frame_step5", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 5); }},
              {"motion_anchor_orientation_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1); }},
              {"motion_joint_positions_10frame_step5", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 5); }},
              {"motion_joint_velocities_10frame_step5", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 5); }},
              {"motion_joint_positions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1); }},
              {"motion_joint_velocities_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1); }},
              {"motion_joint_positions_3frame_step1", 87, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 3, 1); }},
              {"motion_joint_velocities_3frame_step1", 87, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 3, 1); }},
              {"motion_joint_positions_lowerbody_10frame_step5", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 5, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_velocities_lowerbody_10frame_step5", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 5, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_positions_lowerbody_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_velocities_lowerbody_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_positions_wrists_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_positions_wrists_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 2, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_velocities_wrists_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_positions_5frame_step5", 145, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 5, 5); }},
              {"motion_joint_velocities_5frame_step5", 145, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 5, 5); }},
              // SMPL data gathering (dimensions assume 24 joints, 21 poses - adjust based on actual data)
              {"smpl_joints", 72, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 1, 1); }},  // 24*3
              {"smpl_joints_5frame_step5", 360, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 5, 5); }},  // 24*3*5
              {"smpl_joints_10frame_step5", 720, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 10, 5); }},  // 24*3*10
              {"smpl_joints_10frame_step1", 720, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 10, 1); }},  // 24*3*10
              {"smpl_joints_lower_10frame_step1", 270, [this](std::vector<double>& buf, size_t offset) {return GatherMotionSmplJointsMultiFrame(buf, offset, 10, 1, {0,1,2,4,5,7,8,10,11}); }},  // 9*3*10 lower body joints
              {"smpl_joints_2frame_step1", 144, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 2, 1); }},  // 24*3*10
              {"smpl_pose", 63, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 1, 1); }},  // 21*3
              {"smpl_pose_5frame_step5", 315, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 5, 5); }},  // 21*3*5
              {"smpl_pose_10frame_step5", 630, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 5); }},  // 21*3*10
              {"smpl_pose_10frame_step1", 630, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 1); }},  // 21*3*10
              {"smpl_elbow_wrist_poses_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 1, {17, 18, 19, 20}); }},  // 4*3*10
              {"smpl_root_z_10frame_step1", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 1); }},
              {"smpl_anchor_orientation_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1); }},
              {"smpl_anchor_orientation_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 2, 1); }},
              // VR 3-point data gathering
              {"vr_3point_local_target", 9, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointPosition(buf, offset); }},
              // NOTE: In the TELEOP time, the vr_3point_local_target == vr_3point_local_target_compliant
              {"vr_3point_local_target_compliant", 9, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointPosition(buf, offset); }},
              {"vr_3point_local_orn_target", 12, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointOrientation(buf, offset); }},
              {"vr_3point_compliance", 3, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointCompliance(buf, offset); }},
              // VR 5-point data gathering
              {"vr_5point_local_target", 15, [this](std::vector<double>& buf, size_t offset) { return GatherVR5PointPosition(buf, offset); }},
              {"vr_5point_local_orn_target", 20, [this](std::vector<double>& buf, size_t offset) { return GatherVR5PointOrientation(buf, offset); }},
              // History robot state gathering
              {"base_angular_velocity", 3, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 1, 1); }},
              {"body_joint_positions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 1, 1); }},
              {"body_joint_velocities", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 1, 1); }},
              {"last_actions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 1, 1); }},
              {"gravity_dir", 3, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 1, 1); }},
              {"his_body_joint_positions_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 4, 1); }},
              {"his_body_joint_velocities_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 4, 1); }},
              {"his_last_actions_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 4, 1); }},
              {"his_base_angular_velocity_4frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 4, 1); }},
              {"his_gravity_dir_4frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 4, 1); }},
              {"his_body_joint_positions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 10, 1); }},
              {"his_body_joint_velocities_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 10, 1); }},
              {"his_last_actions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 10, 1); }},
              {"his_base_angular_velocity_10frame_step1", 30, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 10, 1); }},
              {"his_gravity_dir_10frame_step1", 30, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 10, 1); }}};
    }
    
    // Initialize observation functions
    void InitializeObservationFunctions() {
      // Get all available observations from registry (single source of truth!)
      auto registry = GetObservationRegistry();
      
      // =========================================================================
      // Initialize policy observations
      // =========================================================================
      active_obs_functions_.clear();
      size_t current_offset = 0;
      
      for (const auto& config : obs_config_) {
        if (!config.enabled) continue;
        
        // Find observation in registry
        auto registry_it = std::find_if(registry.begin(), registry.end(),
          [&config](const ObservationRegistry& obs) {
            return obs.name == config.name;
          });
        
        if (registry_it == registry.end()) {
          std::cerr << "✗ Error: Unknown observation function '" << config.name << "' in configuration!" << std::endl;
          throw std::runtime_error("Invalid observation configuration: unknown function '" + config.name + "'");
        }
        
        // Get dimension directly from registry entry
        size_t dimension = registry_it->dimension;
        if (dimension == 0) {
          std::cerr << "✗ Error: Observation function '" << config.name << "' has invalid dimension!" << std::endl;
          throw std::runtime_error("Invalid observation configuration: zero dimension for '" + config.name + "'");
        }
        
        // Add to active functions (get function directly from registry)
        active_obs_functions_.emplace_back(config.name, registry_it->function, current_offset, dimension);
        current_offset += dimension;
      }
      
      // Validate total dimension matches model input
      size_t model_input_dim = policy_engine_->GetInputDimension();
      if (current_offset != model_input_dim) {
        std::cerr << "✗ Error: Total observation dimension (" << current_offset 
                  << ") doesn't match model input dimension (" << model_input_dim << ")!" << std::endl;
        throw std::runtime_error("Observation dimension mismatch");
      }
      
      // =========================================================================
      // Initialize encoder observations (only if encoder is actually loaded)
      // =========================================================================
      active_encoder_obs_functions_.clear();
      if (encoder_engine_ && encoder_engine_->IsInitialized() && encoder_config_.dimension > 0 && !encoder_config_.encoder_observations.empty()) {
        std::cout << "Initializing encoder observations..." << std::endl;
        size_t encoder_offset = 0;
        size_t encoder_input_size = encoder_engine_->GetInputDimension();
        
        for (const auto& config : encoder_config_.encoder_observations) {
          if (!config.enabled) continue;
          
          // Find observation in registry
          auto registry_it = std::find_if(registry.begin(), registry.end(),
            [&config](const ObservationRegistry& obs) {
              return obs.name == config.name;
            });
          
          if (registry_it == registry.end()) {
            std::cerr << "✗ Error: Unknown encoder observation function '" << config.name << "'!" << std::endl;
            throw std::runtime_error("Invalid encoder observation: unknown function '" + config.name + "'");
          }
          
          size_t dimension = registry_it->dimension;
          if (dimension == 0) {
            std::cerr << "✗ Error: Encoder observation '" << config.name << "' has invalid dimension!" << std::endl;
            throw std::runtime_error("Invalid encoder observation: zero dimension for '" + config.name + "'");
          }
          
          // Add to active encoder functions
          active_encoder_obs_functions_.emplace_back(config.name, registry_it->function, encoder_offset, dimension);
          encoder_offset += dimension;
        }
        
        // Validate encoder input dimension matches expected
        if (encoder_offset != encoder_input_size) {
          std::ostringstream oss;
          oss << "Encoder observation dimension mismatch: got " << encoder_offset
              << ", expected " << encoder_input_size;
          throw std::runtime_error(oss.str());
        }
        
        std::cout << "✓ Initialized " << active_encoder_obs_functions_.size() 
                  << " encoder observations (total dim: " << encoder_offset << ")" << std::endl;
      }
      
      // =========================================================================
      // Check if vr_3point_compliance observation is enabled (in main or encoder obs)
      // This affects whether keyboard compliance controls (g/h/b/v) have any effect
      // =========================================================================
      has_vr_3point_compliance_obs_ = false;
      
      // Check main observations
      for (const auto& obs : active_obs_functions_) {
        if (obs.name == "vr_3point_compliance") {
          has_vr_3point_compliance_obs_ = true;
          break;
        }
      }
      
      // Check encoder observations if not found in main
      if (!has_vr_3point_compliance_obs_) {
        for (const auto& obs : active_encoder_obs_functions_) {
          if (obs.name == "vr_3point_compliance") {
            has_vr_3point_compliance_obs_ = true;
            break;
          }
        }
      }
    }
    
    // Log observation configuration
    void LogObservationConfiguration() {
      std::cout << "\n=== Observation Configuration ===" << std::endl;
      std::cout << "Active policy observations (runtime order):" << std::endl;
      
      size_t total_enabled_dim = 0;
      
      // Show only active observations (clean and focused)
      for (size_t i = 0; i < active_obs_functions_.size(); i++) {
        const auto& active_obs = active_obs_functions_[i];
        
        std::cout << "  " << (i + 1) << ". " 
                  << active_obs.name << " (" << active_obs.dimension << "D)"
                  << " [offset: " << active_obs.offset << "]" << std::endl;
        
        total_enabled_dim += active_obs.dimension;
      }
      
      std::cout << "\nPolicy Summary:" << std::endl;
      std::cout << "  Total dimension: " << total_enabled_dim << std::endl;
      size_t model_input_dim = policy_engine_->GetInputDimension();
      std::cout << "  Model dimension: " << model_input_dim << std::endl;
      
      if (total_enabled_dim == model_input_dim) {
        std::cout << "  ✓ Dimension match: Configuration is valid!" << std::endl;
      } else {
        std::ostringstream oss;
        oss << "Policy observation dimension mismatch: total_enabled_dim=" << total_enabled_dim
            << ", model_input_dim=" << model_input_dim
            << ", difference=" << (int64_t)model_input_dim - (int64_t)total_enabled_dim;
        throw std::runtime_error(oss.str());
      }
      
      // Show encoder observations if enabled
      if (!active_encoder_obs_functions_.empty()) {
        std::cout << "\n--- Encoder Observations ---" << std::endl;
        size_t total_encoder_dim = 0;
        
        for (size_t i = 0; i < active_encoder_obs_functions_.size(); i++) {
          const auto& active_obs = active_encoder_obs_functions_[i];
          
          std::cout << "  " << (i + 1) << ". " 
                    << active_obs.name << " (" << active_obs.dimension << "D)"
                    << " [offset: " << active_obs.offset << "]" << std::endl;
          
          total_encoder_dim += active_obs.dimension;
        }
        
        std::cout << "\nEncoder Summary:" << std::endl;
        std::cout << "  Total dimension: " << total_encoder_dim << std::endl;
        if (encoder_engine_ && encoder_engine_->IsInitialized()) {
          size_t encoder_input_size = encoder_engine_->GetInputDimension();
          std::cout << "  Model dimension: " << encoder_input_size << std::endl;
          
          if (total_encoder_dim == encoder_input_size) {
            std::cout << "  ✓ Dimension match: Encoder configuration is valid!" << std::endl;
          } else {
            std::ostringstream oss;
            oss << "Encoder configuration dimension mismatch: total_encoder_dim=" << total_encoder_dim
                << ", encoder_model_input_dim=" << encoder_input_size;
            throw std::runtime_error(oss.str());
          }
        }
      }
      
      std::cout << "================================\n" << std::endl;
    }
    
    // Gather observations (simplified - functions read from internal state)
    bool GatherObservations() {
      // Clear observation buffer
      std::fill(obs_buffer_.begin(), obs_buffer_.end(), 0.0);
      
      // Process observations using pre-built active functions (no map lookup needed!)
      for (const auto& active_obs : active_obs_functions_) {
        bool success = active_obs.function(obs_buffer_, active_obs.offset);
        
        if (!success) {
          std::cout << "Error: Failed to gather observation: " << active_obs.name << std::endl;
          return false;
        }
      }
      
      return true;
    }

    // Gather encoder observations (populate encoder input buffer)
    // Observation functions read from internal state (state_logger_, current_motion_, etc.)
    // Mode-specific optimization: Only compute observations required for current encoder mode
    bool GatherEncoderObservations() {
      if (!is_using_encoder_ || active_encoder_obs_functions_.empty()) {
        return true;  // Nothing to do
      }

      // Get reference to encoder's internal input buffer
      auto& encoder_input_buffer = encoder_engine_->GetInputBuffer();
      
      // Clear encoder observation buffer (safe default for unused observations)
      std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0.0);
      
      // Store the intended encoder mode for warning purposes
      int intended_encoder_mode = current_motion_->GetEncodeMode();
      
      // Build list of modes to try (start with current mode, then all others as fallback)
      std::vector<int> modes_to_try;
      
      if (!encoder_config_.encoder_modes.empty()) {
        // Start with intended mode if valid
        if (current_motion_->GetEncodeMode() >= 0) {
          modes_to_try.push_back(current_motion_->GetEncodeMode());
        }
        
        // Add all other modes as fallbacks
        for (const auto& mode_config : encoder_config_.encoder_modes) {
          if (mode_config.mode_id != current_motion_->GetEncodeMode()) {
            modes_to_try.push_back(mode_config.mode_id);
          }
        }
      }
      
      // If no modes configured, try without mode filter
      if (modes_to_try.empty()) {
        modes_to_try.push_back(-1);  // -1 means no mode filter (all observations required)
      }
      
      // Try each mode until one succeeds
      for (size_t attempt = 0; attempt < modes_to_try.size(); ++attempt) {
        int mode_to_try = modes_to_try[attempt];
        current_motion_->SetEncodeMode(mode_to_try);
        
        // Determine which observations to gather based on encoder mode
        std::vector<std::string> required_observations;
        bool use_mode_filter = false;
        std::string mode_name;
        
        // Check if mode-specific filtering is configured
        if (!encoder_config_.encoder_modes.empty() && current_motion_->GetEncodeMode() >= 0) {
          // Find the current mode configuration
          for (const auto& mode_config : encoder_config_.encoder_modes) {
            if (mode_config.mode_id == current_motion_->GetEncodeMode()) {
              required_observations = mode_config.required_observations;
              use_mode_filter = true;
              mode_name = mode_config.name;
              
              // Only log when mode changes to avoid spam
              if (current_motion_->GetEncodeMode() != last_logged_encoder_mode_) {
                std::cout << "[Mode Filter] Switched to mode '" << mode_config.name << "' (ID=" << current_motion_->GetEncodeMode() 
                          << ") with " << required_observations.size() << " required observations" << std::endl;
                last_logged_encoder_mode_ = current_motion_->GetEncodeMode();
              }
              break;
            }
          }
        }
        
        // Process encoder observations using pre-built active functions
        bool all_success = true;
        for (const auto& active_obs : active_encoder_obs_functions_) {
          // Check if this observation is required for current mode
          bool is_required = !use_mode_filter;  // If no mode filter, all are required
          
          if (use_mode_filter) {
            // Check if observation is in required list
            is_required = std::find(required_observations.begin(), required_observations.end(), 
                                    active_obs.name) != required_observations.end();
          }
          
          if (is_required) {
            // Compute the observation
            bool success = active_obs.function(encoder_obs_buffer_, active_obs.offset);
            
            if (!success) {
              std::cerr << "✗ Error: Failed to gather encoder observation: " << active_obs.name 
                        << " for mode " << current_motion_->GetEncodeMode() 
                        << (!mode_name.empty() ? " ('" + mode_name + "')" : "") << std::endl;
              all_success = false;
              break;
            }
          }
          // else: Not required for this mode - leave as zero (already initialized)
          //       This saves computation time for expensive observations
        }
        
        // Check if this mode succeeded
        if (all_success) {
          // Success! Copy encoder observations to encoder input buffer (double -> float)
          for (size_t i = 0; i < encoder_input_buffer.size() && i < encoder_obs_buffer_.size(); ++i) {
            encoder_input_buffer[i] = static_cast<float>(encoder_obs_buffer_[i]);
          }
          
          // Warn if we had to switch from intended mode
          if (current_motion_->GetEncodeMode() != intended_encoder_mode && attempt > 0) {
            std::cerr << "⚠ Warning: Intended encoder mode " << intended_encoder_mode 
                      << " failed, successfully switched to fallback mode " << current_motion_->GetEncodeMode() 
                      << (!mode_name.empty() ? " ('" + mode_name + "')" : "") << std::endl;
          }
          
          return true;
        } else {
          // This mode failed - warn if it was the intended mode
          if (current_motion_->GetEncodeMode() == intended_encoder_mode && attempt == 0) {
            std::cerr << "⚠ Warning: Intended encoder mode " << intended_encoder_mode 
                      << (!mode_name.empty() ? " ('" + mode_name + "')" : "")
                      << " failed, trying fallback modes..." << std::endl;
          }
          
          // Clear buffer and try next mode
          std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0.0);
        }
      }
      
      // All modes failed
      std::cerr << "✗ Error: All available encoder modes failed to gather observations" << std::endl;
      return false;
    }



  public:
    OperatorState operator_state;

    G1Deploy(
      std::string networkInterface,
      std::string model_file_path,
      std::string motion_data_path,
      bool disable_crc_check = false,
      std::string obs_config_path = "",
      std::string encoder_file_path = "",
      std::string planner_file_path = "",
      std::string target_motion_file_path = "",
      std::string planner_motion_file_path = "",
      std::string policy_input_file_path = "",
      std::string input_type = "keyboard",
      std::string output_type = "zmq",
      std::string record_input_file_path = "",
      std::string playback_input_file_path = "",
      bool planner_fp16 = false,
      bool policy_fp16 = false,
      std::string logs_dir = "",
      bool enable_csv_logs = false,
      std::string zmq_host = "localhost",
      int zmq_port = 5556,
      std::string zmq_topic = "pose",
      bool zmq_conflate = false,
      bool zmq_verbose = false,
      int zmq_out_port = 5557,
      std::string zmq_out_topic = "g1_debug",
      bool enable_motion_recording = false,
      std::array<double, 3> initial_compliance = {0.05, 0.05, 0.0},
      double initial_max_close_ratio = 1.0)
      : time_(0.0),
        publish_dt_(0.002),
        control_dt_(0.02),
        planner_dt_(0.1),
        input_dt_(0.01),
        duration_(3.0),
        counter_(0),
        mode_pr_(Mode::PR),
        mode_machine_(0),
        disable_crc_check_(disable_crc_check),
        program_state_(ProgramState::INIT),
        last_action {0.0},
        last_left_hand_action {0.0},
        last_right_hand_action {0.0},
        enable_motion_recording_(enable_motion_recording),
        initial_vr_3point_compliance_(initial_compliance),
        initial_max_close_ratio_(initial_max_close_ratio),
        //env(ORT_LOGGING_LEVEL_WARNING, "G1Deploy"),
        model_path(model_file_path),
        planner_path(planner_file_path) {
      
      // Initialize ChannelFactory
      ChannelFactory::Instance()->Init(0, networkInterface);

      // Initialize Dex3 hands (ChannelFactory already initialized above)
      dex3_hands_.initialize("");

      audio_thread_ = std::make_unique<AudioThread>();

      if(!target_motion_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(target_motion_file_path);
        f.close();

        // open file in append mode:
        target_motion_file_ = std::make_unique<std::ofstream>(target_motion_file_path, std::ios::app);
      }

      if(!planner_motion_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(planner_motion_file_path);
        f.close();

        // open file in append mode:
        planner_motion_file_ = std::make_unique<std::ofstream>(planner_motion_file_path, std::ios::app);
      }

      if(!policy_input_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(policy_input_file_path);
        f.close();

        // open file in append mode:
        policy_input_file_ = std::make_unique<std::ofstream>(policy_input_file_path, std::ios::app);
      }

      if(!record_input_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(record_input_file_path);
        f.close();

        // open file in append mode:
        record_input_file_ = std::make_unique<std::ofstream>(record_input_file_path, std::ios::app);
        std::cout << "Initialized record input file: " << record_input_file_path << std::endl;
      }

      if(!playback_input_file_path.empty())
      {
        // open file in append mode:
        playback_input_file_ = std::make_unique<std::ifstream>(playback_input_file_path);
        std::cout << "Initialized playback input file: " << playback_input_file_path << std::endl;
      }

      movement_state_buffer_.SetData(MovementState());

      // Initialize planner motion state as shared_ptr
      planner_motion_ = std::make_shared<MotionSequence>();
      planner_motion_->ReserveCapacity(1500, 29, 1, 1, 0, 0);
      planner_motion_->timesteps = 0;
      planner_motion_->name = "planner_motion";
      // try to shutdown motion control-related service
      msc_ = std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
      msc_->SetTimeout(5.0f);
      msc_->Init();
      std::string form, name;
      while (msc_->CheckMode(form, name), !name.empty()) {
        if (msc_->ReleaseMode()) std::cout << "Failed to switch to Release Mode\n";
        sleep(5);
      }

      // create publisher
      lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
      lowcmd_publisher_->InitChannel();
      // create subscriber
      lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
      lowstate_subscriber_->InitChannel(std::bind(&G1Deploy::LowStateHandler, this, std::placeholders::_1), 1);
      imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
      imutorso_subscriber_->InitChannel(std::bind(&G1Deploy::imuTorsoHandler, this, std::placeholders::_1), 1);
      // Load motion data
      if (motion_reader_.ReadFromCSV(motion_data_path)) {
        if (!motion_reader_.motions.empty()) {
          std::cout << "✓ Motion data loaded successfully!" << std::endl;
          // motion_reader_.PrintSummary();
          motion_reader_.current_motion_index_ = 0;
          std::string motion_name;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
            current_frame_ = 0;
            motion_name = current_motion_->name;
          }
          operator_state.play = false;
          std::cout << "Started with motion: " << motion_name << " (paused at frame 0)" << std::endl;
        } else {
          std::cout << "✗ Error: Motion directory found but no valid motions loaded from " << motion_data_path
                    << std::endl;
          std::cout << "Check that motion folders contain valid CSV files." << std::endl;
          std::cout << "Control system cannot function without motion data!" << std::endl;
          std::cout << "Exiting..." << std::endl;
          throw std::runtime_error("No valid motions found in motion data directory");
        }
      } else {
        std::cout << "✗ Error: Could not access motion data from " << motion_data_path << std::endl;
        std::cout << "Make sure the path exists and contains motion folders." << std::endl;
        std::cout << "Control system cannot function without motion data!" << std::endl;
        std::cout << "Exiting..." << std::endl;
        throw std::runtime_error("Failed to load motion data");
      }
      
      // Initialize control policy
      policy_engine_ = std::make_unique<PolicyEngine>();
      
      if (!policy_engine_->Initialize(model_path, policy_fp16)) {
        throw std::runtime_error("Failed to initialize control policy from: " + model_path);
      }
      
      // Initialize observation buffer with correct size (zero-initialized)
      size_t obs_dim = policy_engine_->GetInputDimension();
      obs_buffer_.resize(obs_dim, 0.0);
      
      // Capture CUDA graph for optimized execution
      if (!policy_engine_->CaptureGraph()) {
        throw std::runtime_error("Failed to capture control policy CUDA graph");
      }
      
      std::cout << "✓ Policy model loaded successfully!" << std::endl;

      // Load observation configuration FIRST (before encoder/planner initialization)
      std::cout << "Loading observation configuration..." << std::endl;
      FullObservationConfig full_obs_config;
      if (!obs_config_path.empty()) {
        std::cout << "Using observation config file: " << obs_config_path << std::endl;
        full_obs_config = ObservationConfigParser::ParseFullConfig(obs_config_path);
      } else {
        std::cout << "Using default observation configuration" << std::endl;
        full_obs_config.observations = ObservationConfigParser::ParseConfig();
      }
      
      // Check if config parsing failed (empty observations returned)
      if (full_obs_config.observations.empty()) {
        throw std::runtime_error("Failed to parse observation configuration - check config file for errors");
      }
      
      obs_config_ = full_obs_config.observations;
      encoder_config_ = full_obs_config.encoder;
      
      // Initialize token buffer size from encoder config
      if (encoder_config_.dimension > 0) {
        token_state_data_.resize(encoder_config_.dimension, 0.0);
        std::cout << "✓ Token state buffer initialized (size: " << encoder_config_.dimension << ")" << std::endl;
      }
      
      // =========================================================================
      // Initialize encoder engine
      // =========================================================================
      // Note: Parser already validates that if token_state is enabled, encoder_config has dimension > 0
      // So if dimension <= 0 here, it means token_state is disabled - just ignore encoder_file_path
      if (!encoder_file_path.empty() && encoder_config_.dimension > 0) {
        std::cout << "Initializing encoder..." << std::endl;
        encoder_engine_ = std::make_unique<EncoderEngine>();
        
        if (!encoder_engine_->Initialize(encoder_file_path, encoder_config_.use_fp16)) {
          throw std::runtime_error("Failed to initialize encoder engine from: " + encoder_file_path);
        }
        
        // Validate encoder output dimension matches config
        if (encoder_engine_->GetTokenDimension() != static_cast<size_t>(encoder_config_.dimension)) {
          std::cerr << "⚠ Warning: Encoder model output dimension (" << encoder_engine_->GetTokenDimension() 
                    << ") doesn't match config dimension (" << encoder_config_.dimension << ")" << std::endl;
        }
        
        // Initialize encoder observation buffer with correct size
        size_t encoder_input_size = encoder_engine_->GetInputDimension();
        encoder_obs_buffer_.resize(encoder_input_size, 0.0);

        // Capture CUDA graph for optimized execution
        if (!encoder_engine_->CaptureGraph()) {
          throw std::runtime_error("Failed to capture encoder CUDA graph");
        }
        
        std::cout << "✓ Encoder model loaded successfully!" << std::endl;
        is_using_encoder_ = true;
        initial_encoder_mode_ = 0;  // Encoder available, default to mode 0.
      } else {
        if (encoder_config_.dimension > 0) {
          std::cout << "Encoder config found but no encoder file provided - tokens can be set externally" << std::endl;
          initial_encoder_mode_ = -1;
        } else {
          std::cout << "No encoder configured" << std::endl;
          initial_encoder_mode_ = -2;
        }
        is_using_encoder_ = false;
      }
      
      // =========================================================================
      // Initialize encode_mode for all loaded reference motions and planner motion
      // =========================================================================
      // Set the encode_mode for all loaded reference motions based on encoder availability
      std::cout << "Setting encode_mode for " << motion_reader_.motions.size() << " loaded reference motions..." << std::endl;
      for (auto& motion : motion_reader_.motions) {
        motion->SetEncodeMode(initial_encoder_mode_);
        std::cout << "  Motion '" << motion->name << "' encode_mode set to: " << initial_encoder_mode_ << std::endl;
      }
      
      // Set encode_mode for planner motion
      planner_motion_->SetEncodeMode(initial_encoder_mode_);
      std::cout << "Planner motion encode_mode set to: " << initial_encoder_mode_ << std::endl;

      // Initialize planner
      if (!planner_path.empty()) {
        PlannerConfig planner_config;
        planner_config.model_path = planner_path;
        if (planner_path.find("V0") != std::string::npos)
        {
          planner_config.version = 0;
        }
        else if (planner_path.find("V1") != std::string::npos)
        {
          planner_config.version = 1;
        }
        else if (planner_path.find("V2") != std::string::npos)
        {
          planner_config.version = 2;
        }
        else
        {
          std::cout << "Unsupported planner version: " << planner_path << std::endl;
          throw std::runtime_error("Unsupported planner version: " + planner_path);
        }
        planner_ = std::make_unique<LocalMotionPlannerTensorRT>(planner_fp16, 0, planner_config);
      }
      
      // Initialize observation function map
      InitializeObservationFunctions();
      
      // Log observation configuration details
      LogObservationConfiguration();

      // Prepare robot configuration for data collection (after all initialization is complete)
      std::map<std::string, std::variant<std::string, int, double, bool>> robot_config;
      robot_config["model_path"] = model_path;
      robot_config["reference_motion_path"] = motion_data_path;
      robot_config["planner_path"] = planner_path.empty() ? "none" : planner_path;
      robot_config["obs_config_path"] = obs_config_path.empty() ? "none" : obs_config_path;
      robot_config["encoder_file"] = encoder_file_path.empty() ? "none" : encoder_file_path;
      robot_config["control_frequency"] = 1.0 / control_dt_;
      robot_config["planner_frequency"] = 1.0 / planner_dt_;
      robot_config["is_using_encoder"] = is_using_encoder_;
      robot_config["policy_fp16"] = policy_fp16;
      robot_config["planner_fp16"] = planner_fp16;

      // Initialize state logger with complete robot configuration
      try {
        std::string resolved_logs_dir = logs_dir;
        // dt is control_dt_; pass complete robot_config to constructor
        state_logger_ = std::make_unique<StateLogger>(resolved_logs_dir, 10000, G1_NUM_MOTOR, G1_NUM_MOTOR, 
                                                       control_dt_, enable_csv_logs, robot_config);
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize state logger: " << e.what() << std::endl;
        std::cerr << "State logger is required for operation. Exiting..." << std::endl;
        throw;  // Re-throw to stop program initialization
      }

      // Initialize input interface based on type
      if (input_type == "gamepad") {
        input_interface_ = std::make_unique<unitree::common::Gamepad>();
        std::cout << "Initialized gamepad input interface" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      } 
      else if (input_type == "gamepad_manager") {
        input_interface_ = std::make_unique<GamepadManager>(zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose);
        std::cout << "Initialized demo gamepad manager input interface" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Verbose: " << (zmq_verbose ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "manager") {
        input_interface_ = std::make_unique<InterfaceManager>(
          zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose
        );
        std::cout << "Initialized interface manager (Shift+1/2/3/4 [! @ # $] to switch: keyboard, gamepad, zmq"
#if HAS_ROS2
                  " , ros2"
#endif
                  ")" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "zmq") {
        input_interface_ = std::make_unique<ZMQEndpointInterface>(
          zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose
        );
        std::cout << "Initialized ZMQ endpoint interface" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "zmq_manager") {
        input_interface_ = std::make_unique<ZMQManager>(
          zmq_host, zmq_port, zmq_topic, "command", "planner", zmq_conflate, zmq_verbose
        );
        std::cout << "Initialized ZMQ manager" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Pose topic: " << zmq_topic << std::endl;
        std::cout << "  Command topic: command" << std::endl;
        std::cout << "  Planner topic: planner" << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
#if HAS_ROS2
      else if (input_type == "ros2") {
        // ROS2 input interface will be initialized without encoder mode parameter
        input_interface_ = std::make_unique<ROS2InputHandler>(true, "g1_deploy_ros2_handler");
        std::cout << "Initialized ROS2 input interface" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
#endif
      else {
        input_interface_ = std::make_unique<SimpleKeyboard>();
        std::cout << "Initialized keyboard input interface (default)" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }

      // Set initial VR 3-point compliance values for all input interfaces
      // These are keyboard-controlled: g/h for left hand, b/v for right hand
      if (input_interface_) {
        input_interface_->SetVR3PointCompliance(initial_vr_3point_compliance_);
        // Set initial max close ratio for hands (keyboard-controlled: X/C keys)
        input_interface_->SetMaxCloseRatio(initial_max_close_ratio_);
        dex3_hands_.SetMaxCloseRatio(initial_max_close_ratio_);
        std::cout << "[INFO] Initial VR 3-point compliance: ["
                  << initial_vr_3point_compliance_[0] << ", "
                  << initial_vr_3point_compliance_[1] << ", "
                  << initial_vr_3point_compliance_[2] << "]" << std::endl;
        std::cout << "[INFO] Initial hand max close ratio: " << initial_max_close_ratio_ 
                  << " (1.0 = full closure allowed, 0.2 = limited)" << std::endl;
        std::cout << "[INFO] Keyboard controls: g/h = left hand +/- 0.1, b/v = right hand +/- 0.1 (range: 0.01-0.5)" << std::endl;
        std::cout << "[INFO] Keyboard controls: x/c = hand max close ratio +/- 0.1 (range: 0.2-1.0)" << std::endl;
        
        // Info message about compliance observation status
        if (!has_vr_3point_compliance_obs_) {
          std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
          std::cout << "│  Compliance control: DISABLED                                               │" << std::endl;
          std::cout << "│  Policy does not observe 'vr_3point_compliance'.                            │" << std::endl;
          std::cout << "│  Keyboard controls (g/h/b/v) will be ignored for this policy.               │" << std::endl;
          std::cout << "│  To enable, use a compliance-aware policy that observes it natively.        │" << std::endl;
          std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n" << std::endl;
        } else {
          std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
          std::cout << "│  Compliance control: ENABLED                                                │" << std::endl;
          std::cout << "│  Policy observes 'vr_3point_compliance'. Keyboard controls (g/h/b/v) work.  │" << std::endl;
          std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n" << std::endl;
        }
      }

      // Initialize output interfaces based on type (now that state_logger_ is ready)
      // Supports multiple simultaneous outputs with --output-type all
      bool create_zmq = (output_type == "zmq" || output_type == "all");
      bool create_ros2 = false;
#if HAS_ROS2
      create_ros2 = (output_type == "ros2" || output_type == "all");
#endif

      if (create_zmq) {
        output_interfaces_.push_back(std::make_unique<ZMQOutputHandler>(*state_logger_, zmq_out_port, zmq_out_topic));
        std::cout << "Initialized ZMQ output interface" << std::endl;
      }

#if HAS_ROS2
      if (create_ros2) {
        auto ros2_handler = std::make_unique<ROS2OutputHandler>(*state_logger_, "g1_output_handler");
        std::cout << "Initialized ROS2 output interface" << std::endl;
        // Publish robot config so subscribers can receive it before control loop starts
        ros2_handler->publish_config();
        output_interfaces_.push_back(std::move(ros2_handler));
      }
#endif

      if (output_interfaces_.empty() && output_type != "all") {
        std::cout << "Unknown output type '" << output_type << "' - no output will be published" << std::endl;
      } else {
        std::cout << "Total output interfaces initialized: " << output_interfaces_.size() << std::endl;
      }

      // create threads
      input_thread_ptr_ = CreateRecurrentThreadEx("Input", UT_CPU_ID_NONE, input_dt_ * 1e6, &G1Deploy::Input, this);
      command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, publish_dt_ * 1e6,
                                                    &G1Deploy::LowCommandWriter, this);
      control_thread_ptr_ =
          CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, control_dt_ * 1e6, &G1Deploy::Control, this);
      
      if (planner_) {
        planner_thread_ptr_ =
          CreateRecurrentThreadEx("planner", UT_CPU_ID_NONE, planner_dt_ * 1e6, &G1Deploy::Planner, this);
      }
          
      SetThreadPriority();
    }

    ~G1Deploy()
    {
      // CUDA resources are now cleaned up by the PolicyEngine and planner classes automatically
    }

    void SetThreadPriority() {
      struct sched_param param;
      param.sched_priority = sched_get_priority_max(SCHED_FIFO);
      pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    /// DDS callback: receives a 500 Hz LowState message from the robot SDK.
    /// Validates CRC (unless disabled for MuJoCo sim) and stores the result
    /// in the thread-safe low_state_buffer_.
    void LowStateHandler(const void* message) {
      LowState_ low_state = *(const LowState_*)message;

      // Only perform CRC check if not disabled (for MuJoCo simulation compatibility)
      if (!disable_crc_check_) {
        uint32_t received_crc = low_state.crc();
        uint32_t calculated_crc = Crc32Core((uint32_t*)&low_state, (sizeof(LowState_) >> 2) - 1);

        if (received_crc != calculated_crc) {
          static int crc_error_count = 0;
          crc_error_count++;
          std::cout << "[ERROR] CRC Error #" << crc_error_count << " - Received: 0x" << std::hex << received_crc
                    << ", Calculated: 0x" << calculated_crc << ", Size: " << sizeof(LowState_) << " bytes" << std::dec
                    << std::endl;
          return;
        }
      }

      low_state_buffer_.SetData(low_state);

      // update mode machine
      if (mode_machine_ != low_state.mode_machine()) {
        if (mode_machine_ == 0) std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
        mode_machine_ = low_state.mode_machine();
      }
    }

    /// DDS callback: receives secondary (torso) IMU data.
    void imuTorsoHandler(const void* message) {
      IMUState_ imu_torso = *(const IMUState_*)message;
      imu_torso_buffer_.SetData(imu_torso);
    }

    /**
     * @brief Command-writer thread body (500 Hz).
     *
     * Reads the latest MotorCommand from motor_command_buffer_, packs it
     * into a LowCmd_ DDS message with CRC, and publishes via DDS.
     * Also publishes Dex3 hand commands at the same cadence.
     */
    void LowCommandWriter() {
      LowCmd_ dds_low_command;
      dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
      dds_low_command.mode_machine() = mode_machine_;

      const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetDataWithTime().data;
      if (mc) {
        for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
          dds_low_command.motor_cmd().at(i).mode() = 1; // 1:Enable, 0:Disable
          dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
          dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
          dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
          dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
          dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
        }

        dds_low_command.crc() = Crc32Core((uint32_t*)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
        lowcmd_publisher_->Write(dds_low_command);
      }

      // Publish Dex3 hand commands at the same publish cadence
      dex3_hands_.writeOnce();
    }

    /// Gracefully stop all threads and send a damping-only command.
    void Stop() {
      operator_state.stop = true;

      if (control_thread_ptr_) {
        input_thread_ptr_->Wait();
        input_thread_ptr_.reset();
        control_thread_ptr_->Wait();
        control_thread_ptr_.reset();
        command_writer_ptr_->Wait();
        command_writer_ptr_.reset();
        if (planner_thread_ptr_) {
          planner_thread_ptr_->Wait();
          planner_thread_ptr_.reset();
        }
      }
      CreateDampingCommand();
      LowCommandWriter();
      std::cout << "Stop" << std::endl;
    }

    /// Write a zero-torque, damping-only motor command (safe shutdown pose).
    void CreateDampingCommand() {
      MotorCommand motor_command_tmp;
      const std::shared_ptr<const LowState_> ls = low_state_buffer_.GetDataWithTime().data;

      for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.q_target.at(i) = 0.0;
        motor_command_tmp.dq_target.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = 0;
        motor_command_tmp.kd.at(i) = 8;
      }
      motor_command_buffer_.SetData(motor_command_tmp);
    }

    /**
     * @brief INIT state handler: ramp the robot from its current pose to the
     *        default standing angles over `duration_` seconds (linear interpolation).
     *
     * Called at 50 Hz until the ramp completes, at which point the state machine
     * transitions to WAIT_FOR_CONTROL and the Dex3 hands open.
     * @return True once LowState data is available; false if not yet ready.
     */
    bool InitControl() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      if (!ls) {
        return false;
      }
      MotorCommand motor_command_tmp;
      for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.q_target.at(i) = static_cast<float>(default_angles[i]);
        motor_command_tmp.dq_target.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = kps[i];
        motor_command_tmp.kd.at(i) = kds[i];
      }
      time_ += control_dt_;
      if (time_ < duration_) {
        for (int i = 0; i < G1_NUM_MOTOR; i++) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
          double current_pos = ls->motor_state()[i].q();
          motor_command_tmp.q_target.at(i) =
              static_cast<float>(current_pos * (1.0 - ratio) + default_angles[i] * ratio);
        }
        dex3_hands_.close(true);
        dex3_hands_.close(false);
      } else {
        program_state_ = ProgramState::WAIT_FOR_CONTROL;
        dex3_hands_.open(true);
        dex3_hands_.open(false);
        std::cout << "Init Done" << std::endl;
      }
      motor_command_buffer_.SetData(motor_command_tmp);
      return true;
    }

    /// Get current ROS 2 timestamp (seconds) from the first ROS2 output interface, or 0.0 if none.
    double GetRosTimestamp() {
      // Get ROS timestamp from the first ROS2 output interface (0.0 if none available)
      double ros_timestamp = 0.0;
#if HAS_ROS2
      for (auto& output_interface : output_interfaces_) {
        if (output_interface && output_interface->GetType() == OutputInterface::OutputType::ROS2) {
          auto* ros2_handler = dynamic_cast<ROS2OutputHandler*>(output_interface.get());
          if (ros2_handler) {
            ros_timestamp = ros2_handler->GetROSTimestamp();
            break;  // Use timestamp from first ROS2 interface found
          }
        }
      }
#endif
      return ros_timestamp;
    }

    /**
     * @brief Read the latest robot state (IMU + joints + hands) and log it.
     *
     * Reads LowState and torso IMU from their DataBuffers, remaps joint positions
     * from hardware order to IsaacLab order (subtracting default_angles), and
     * calls StateLogger::LogFullState() with the complete snapshot.
     *
     * @return False if LowState or IMU data is missing or a joint velocity
     *         exceeds the safety threshold (35 rad/s).
     */
    bool GatherRobotStateToLogger() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      auto imu_data = imu_torso_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      const std::shared_ptr<const IMUState_> imu_torso = imu_data.data;
      if (!ls || !imu_torso) {
        std::cout << "✗ Error: LowState or IMUState is not available in the middle of the control loop!" << std::endl;
        return false;
      }
      used_low_state_data_ = (low_state_data);
      used_imu_torso_data_ = (imu_data);
      // robot state data
      std::array<double, G1_NUM_MOTOR> body_q = {0.0};
      std::array<double, G1_NUM_MOTOR> body_dq = {0.0};

      auto unitree_joint_state = ls->motor_state();
      for (int i = 0; i < G1_NUM_MOTOR; i++) {
        body_q[i] =
            unitree_joint_state[mujoco_to_isaaclab[i]].q() - default_angles[mujoco_to_isaaclab[i]]; // URDF order
        body_dq[i] = unitree_joint_state[mujoco_to_isaaclab[i]].dq(); // URDF order
        if (body_dq[i] > 35 && !disable_crc_check_) {
          std::cout << "✗ Error: body_dq[" << i << "] = " << body_dq[i] << " > 35."
                    << std::endl;
          return false;
        }
      }

      std::array<double, 4> base_quat = float_to_double<4>(ls->imu_state().quaternion()); // qw, qx, qy, qz
      std::array<double, 3> base_ang_vel = float_to_double<3>(ls->imu_state().gyroscope());
      std::array<double, 3> base_accel = float_to_double<3>(ls->imu_state().accelerometer());

      std::array<double, 4> body_torso_quat = float_to_double<4>(imu_torso->quaternion()); //qw, qx, qy, qz 
      std::array<double, 3> body_torso_ang_vel = float_to_double<3>(imu_torso->gyroscope());
      std::array<double, 3> body_torso_accel = float_to_double<3>(imu_torso->accelerometer());

      // Collect hand states from Dex3 hands
      std::array<double, 7> left_hand_q = {0.0};
      std::array<double, 7> left_hand_dq = {0.0};
      std::array<double, 7> right_hand_q = {0.0};
      std::array<double, 7> right_hand_dq = {0.0};
      
      auto left_hand_state_ptr = dex3_hands_.getState(true);
      if (left_hand_state_ptr) {
        for (int i = 0; i < 7; ++i) {
          left_hand_q[i] = left_hand_state_ptr->motor_state()[i].q();
          left_hand_dq[i] = left_hand_state_ptr->motor_state()[i].dq();
        }
      }
      
      auto right_hand_state_ptr = dex3_hands_.getState(false);
      if (right_hand_state_ptr) {
        for (int i = 0; i < 7; ++i) {
          right_hand_q[i] = right_hand_state_ptr->motor_state()[i].q();
          right_hand_dq[i] = right_hand_state_ptr->motor_state()[i].dq();
        }
      }

      // Log robot state for analysis and debugging
      if (state_logger_) {
        // Get ROS timestamp if using ROS2 output interface (0.0 for non-ROS2 interfaces)
        double ros_timestamp = GetRosTimestamp();
        state_logger_->LogFullState(base_quat, base_ang_vel, base_accel, body_torso_quat, body_torso_ang_vel, body_torso_accel,
                                    std::span(body_q),
                                    std::span(body_dq),
                                    std::span(last_action),
                                    std::span(left_hand_q),
                                    std::span(left_hand_dq),
                                    std::span(right_hand_q),
                                    std::span(right_hand_dq),
                                    std::span(last_left_hand_action),
                                    std::span(last_right_hand_action),
                                    ros_timestamp);
      }
      return true;
    }

    /**
     * @brief Snapshot all input-interface data into local buffers.
     *
     * Called once per control tick **before** GatherObservations().  Copies VR
     * positions / orientations / compliance, hand joints, upper-body targets,
     * and external token state from the active InputInterface into member
     * buffers so that observations, logging, and output all use a consistent
     * snapshot.
     *
     * Also handles the external-token-state fallback logic:
     *  - If encoder is loaded → token_state_data_ will be populated later
     *    by GatherTokenState() during observation gathering.
     *  - If no encoder and external token data arrives (ZMQ/ROS2) → copies it.
     *  - If no encoder and no external data → error recovery depending on
     *    initial_encoder_mode_ (-1 = stop, 0+ = fall back to local encoder).
     */
    bool GatherInputInterfaceData() {
      // Snapshot input interface data into buffers before gathering observations
      // This ensures observations, logging, and output all use the same data
      // some of the data will be overwritten by the motion dataset if we are not using the buffered interface data
      std::tie(has_vr_3point_data_, vr_3point_position_buffer_) = input_interface_->GetVR3PointPosition();
      std::tie(std::ignore, vr_3point_orientation_buffer_) = input_interface_->GetVR3PointOrientation();
      vr_3point_compliance_buffer_ = input_interface_->GetVR3PointCompliance();
      std::tie(has_vr_5point_data_, vr_5point_position_buffer_) = input_interface_->GetVR5PointPosition();
      std::tie(std::ignore, vr_5point_orientation_buffer_) = input_interface_->GetVR5PointOrientation();
      std::tie(has_left_hand_data_, left_hand_joint_buffer_) = input_interface_->GetHandPose(true);
      std::tie(has_right_hand_data_, right_hand_joint_buffer_) = input_interface_->GetHandPose(false);
      std::tie(has_upper_body_data_, upper_body_joint_positions_buffer_) = input_interface_->GetUpperBodyJointPositions();
      std::tie(std::ignore, upper_body_joint_velocities_buffer_) = input_interface_->GetUpperBodyJointVelocities();

      auto last_update_time = input_interface_->GetLastUpdateTime();
      if (last_update_time.has_value()) {
        auto streaming_data_delay = std::chrono::steady_clock::now() - last_update_time.value();
        streaming_data_delay_rolling_stats_.push(static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(streaming_data_delay).count()));

        bool streaming_data_absent = streaming_data_delay > STREAMING_DATA_ABSENT_THRESHOLD;
        streaming_data_absent_debouncer_.update(streaming_data_absent);
      }
      
      audio_thread_->SetCommand(
        AudioCommand{
          .streaming_data_absent = streaming_data_absent_debouncer_.state(),
        });

      // if encoder mode is not -2, we need to get the token state data either from the encoder or the external token state
      if (initial_encoder_mode_ != -2) {
        // Reset the current token_state_data_ to zeros
        token_state_data_.assign(token_state_data_.size(), 0.0);
        // get the external token state from the input interface
        auto [has_ext_token_data, temp_token_state_data] = input_interface_->GetExternalTokenState();
        if (has_ext_token_data && !temp_token_state_data.empty() && !is_using_encoder_) { // if encoder is used, the token state data will be set by the encoder
          // check if the size of the temp_token_state_data is the same as the token_state_data_
          if (temp_token_state_data.size() != token_state_data_.size()) {
            std::cout << "⚠ Warning: The size of the external token state is not the same as what defined in the config file" << std::endl;
            std::cout << "temp_token_state_data.size(): " << temp_token_state_data.size() << std::endl;
            std::cout << "token_state_data_.size(): " << token_state_data_.size() << std::endl;
            std::cout << "No copying happened." << std::endl;
          } else {
            std::copy(temp_token_state_data.begin(), temp_token_state_data.end(), token_state_data_.begin());
          }
        } 
        else if (!has_ext_token_data && !is_using_encoder_) { // if encoder is not used and the external token state is empty
          if (initial_encoder_mode_ == -1) { // if encoder mode is -1, we need to set start control to false and go back to INIT state since we need the token state data
            std::cout << "⚠ Warning: No encoder is loaded and the external token state is empty" << std::endl;
            std::cout << "Set start control to false and go back to INIT state" << std::endl;
            operator_state.start = false;
            program_state_ = ProgramState::INIT;
            time_ = 0.0;
            return false;
          } else { 
            // if encoder mode is not -1, we need to turn on the encoder stop the play 
            // and reinitialize to reference motion at frame 0
            std::cout << "⚠ Warning: No encoder is used and the external token state is empty" << std::endl;
            std::cout << "Turning on the encoder and reinitialize to reference motion at frame 0" << std::endl;
            is_using_encoder_ = true;
            // Reset current motion and frame (protected by mutex)
            {
              std::lock_guard<std::mutex> lock(current_motion_mutex_);
              operator_state.play = false;
              current_frame_ = 0;
              current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
              current_motion_->SetEncodeMode(0);
              reinitialize_heading_ = true;
            }
            // Reset movement state
            movement_state_buffer_.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
          }
        }
      }
      return true;
    }

    /**
     * @brief Run the policy neural network and produce a motor command.
     *
     * Converts the double-precision observation buffer to float, copies it
     * into the PolicyEngine's pinned input buffer, runs TensorRT inference,
     * then maps the action output (IsaacLab order) to a MotorCommand
     * (hardware order) using `g1_action_scale` and `default_angles`.
     */
    bool CreatePolicyCommand() {
      // Convert double observation to float and populate policy's internal input buffer
      auto& obs_buffer_float = policy_engine_->GetInputBuffer();
      for (size_t i = 0; i < obs_buffer_.size(); i++) { 
        obs_buffer_float[i] = static_cast<float>(obs_buffer_[i]); 
      }

      // Run policy inference (handles CPU→GPU transfer, inference, GPU→CPU transfer)
      if (!policy_engine_->Infer()) {
        std::cerr << "✗ Error: Control policy inference failed" << std::endl;
        return false;
      }
      
      // Access actions from control policy's internal buffer (already populated by Infer)
      auto& action_buffer = policy_engine_->GetActionBuffer();
      float* floatarr = action_buffer.data();
      
      MotorCommand motor_command_tmp;
      for (int i = 0; i < G1_NUM_MOTOR; i++) {
        const double action_value = static_cast<double>(floatarr[isaaclab_to_mujoco[i]]) * g1_action_scale[i];
        last_action[i] = static_cast<double>(floatarr[i]);
        motor_command_tmp.q_target.at(i) = static_cast<float>(default_angles[i] + action_value);
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = kps[i];
        motor_command_tmp.kd.at(i) = kds[i];
        motor_command_tmp.dq_target.at(i) = 0.0;
      }
      motor_command_buffer_.SetData(motor_command_tmp);
      return true;
    }

    /**
     * @brief Advance the playback cursor and blend planner output.
     *
     * Called at the end of each control tick.  Two cases:
     *
     * 1. **Planner active**: Checks if a new planner animation is available.
     *    If so, blends it into the existing planner_motion_ using a linear
     *    cross-fade over 8 frames.  Then advances current_frame_ by 1.
     *
     * 2. **Reference-motion mode**: Advances current_frame_ by 1.  Clamps or
     *    resets to 0 when the motion ends (except for "streamed" motions,
     *    which hold the last valid frame while waiting for new data).
     *
     * @return True on success.
     */
    bool CurrentFrameAdvancement() {
      // get current motion and frame from planner when planner is enabled and initialized
      std::lock_guard<std::mutex> motion_lock(current_motion_mutex_);
      if (planner_ && planner_->planner_state_.enabled && planner_->planner_state_.initialized) {
        std::lock_guard<std::mutex> planner_lock(planner_->planner_motion_mutex_);
        
        // Check if the planner thread has finished generating a valid animation:
        if (planner_->motion_available_ && planner_->planner_motion_50hz_.timesteps > 0) {
          
          const auto &planner_motion_gen = planner_->planner_motion_50hz_;
          planner_->motion_available_ = false;

          // special case if this is the first time we're receiving a planner motion:
          bool is_the_first_time = false;
          bool success = false;
          if(planner_motion_->timesteps == 0)
          {
            // step through the frames and copy the animation over:
            for(int f = 0; f < planner_motion_gen.timesteps; ++f)
            {
              std::copy(
                planner_motion_gen.JointPositions(f),
                planner_motion_gen.JointPositions(f) + planner_motion_gen.GetNumJoints(),
                planner_motion_->JointPositions(f)
              );

              std::copy(
                planner_motion_gen.JointVelocities(f),
                planner_motion_gen.JointVelocities(f) + planner_motion_gen.GetNumJoints(),
                planner_motion_->JointVelocities(f)
              );

              planner_motion_->BodyPositions(f)[0] = planner_motion_gen.BodyPositions(f)[0];
              planner_motion_->BodyQuaternions(f)[0] = planner_motion_gen.BodyQuaternions(f)[0];
            }
            planner_motion_->timesteps = planner_motion_gen.timesteps;
            success = true;
            is_the_first_time = true;
          }
          else
          {

            auto fgen = planner_->gen_frame_;

            // Ok, a new planner animation is available. We want to do these things:
            //  * rebase the old planner_motion_ so current_frame_ is at frame 0
            //  * fade in the new planner animation, starting at frame fgen - current_frame_
            //    (or frame 0 if fgen < current_frame_)

            // The newly generated animation starts at frame fgen - current_frame_ on this
            // new timeline, so the total length of the animation will be this:
            auto new_anim_length = fgen - current_frame_ + planner_motion_gen.timesteps;

            // guess we've got to sit this out if things arrived so late that we don't have any frames
            // we can copy over...
            if(new_anim_length > 0)
            {
              // We want to insert a region where the old animation is blended with the new one
              // using a linearly decaying weight. If current_frame_ > fgen, this needs to start
              // immediately, otherwise it should start at at frame fgen - current_frame_:
              auto blend_start_frame = std::max(0, fgen - current_frame_);

              // choosing 8 for the width of the blend region because this is about the size of the
              // context we use for conditioning the planner model
              const int blend_num_frames = 8;

              // step through the frames and blend the animations:
              for(int f = 0; f < new_anim_length; ++f)
              {
                // corresponding frames in the old animation and the new one from the
                // planner that we're blending in:
                int f_old = f + current_frame_;
                int f_new = f_old - fgen;

                // clamp to valid ranges:
                f_old = std::clamp(f_old, 0, planner_motion_->timesteps - 1);
                f_new = std::clamp(f_new, 0, planner_motion_gen.timesteps - 1);
                
                // calculate linearly decaying blend weight:
                double w_new = double(f - blend_start_frame) / blend_num_frames;
                w_new = std::clamp(w_new, 0.0, 1.0);
                double w_old = 1.0 - w_new;

                // blend joint_positions + joint_velocities:
                for(size_t j=0; j < planner_motion_->GetNumJoints(); ++j)
                {
                  planner_motion_->JointPositions(f)[j] = 
                    w_old * planner_motion_->JointPositions(f_old)[j] +
                    w_new * planner_motion_gen.JointPositions(f_new)[j];
                  planner_motion_->JointVelocities(f)[j] = 
                      w_old * planner_motion_->JointVelocities(f_old)[j] +
                      w_new * planner_motion_gen.JointVelocities(f_new)[j];
                }

                // blend the global xyz position:
                for(int j=0; j < 3; ++j)
                {
                  planner_motion_->BodyPositions(f)[0][j] = 
                      w_old * planner_motion_->BodyPositions(f_old)[0][j] +
                      w_new * planner_motion_gen.BodyPositions(f_new)[0][j];
                }
                
                // blend the global rotation quaternion using a slerp:
                std::array<double, 4> q0 = planner_motion_->BodyQuaternions(f_old)[0];
                std::array<double, 4> q1 = planner_motion_gen.BodyQuaternions(f_new)[0];
                planner_motion_->BodyQuaternions(f)[0] = quat_slerp_d(q0, q1, w_new);
              }

              // set the new timestep count and reset the current frame to 0:
              planner_motion_->timesteps = new_anim_length;
              success = true;
            }
          }

          if(success)
          {
            if(planner_motion_file_)
            {
              // log the motion that we just copied/blended over:
              for(int frame = 0; frame < planner_motion_->timesteps; frame++) {
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][0] << ",";
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][1] << ",";
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][2] << ",";

                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][0] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][1] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][2] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][3] << ",";

                for(int i = 0; i < 29; i++) {
                  (*planner_motion_file_) << planner_motion_->JointPositions(frame)[isaaclab_to_mujoco[i]] << ",";
                }

                (*planner_motion_file_) << std::endl;
              }

              // insert a blank line to separate motions:
              (*planner_motion_file_) << std::endl;
            }

            // switch the current motion to planner_motion:
            current_frame_ = 0;
            // Assign shared_ptr directly - planner_motion_ is already a shared_ptr
            current_motion_ = planner_motion_;
            if(is_the_first_time) {
              reinitialize_heading_ = true;
            }
          }
        }
          
        // Frame advancement at control frequency (50Hz) for smooth playback
        if (current_motion_->timesteps > 0 && operator_state.play) {
          int new_frame = current_frame_ + 1;
            if (new_frame >= current_motion_->timesteps) {
              new_frame = current_motion_->timesteps - 1; // Clamp to last frame
            }
          current_frame_ = new_frame;
        }
      }

      // Update motion frame playback for non-planner motion
      bool use_planner_motion = (current_motion_ && current_motion_ == planner_motion_);
      if (!use_planner_motion && current_motion_ && current_motion_->timesteps > 0 && operator_state.play) {
        // Update display motion current frame
        current_frame_++;
        // Check if motion completed (reached the end)
        if (current_motion_->name != "streamed") {
          if (current_frame_ >= current_motion_->timesteps) {
            operator_state.play = false;
            std::cout << "Motion " << current_motion_->name << " completed." << std::endl;
            current_frame_ = 0; // Reset to beginning
            // Total reset: both base quaternion and delta heading
            reinitialize_heading_ = true;
            std::cout << "Reset to frame 0." << std::endl;
          }
        } else {
          if (current_frame_ >= current_motion_->timesteps - saved_frame_for_observation_window_) {
            current_frame_ = current_frame_ - 1;
            std::cout << "Motion " << current_motion_->name << " completed and waiting following motion" << std::endl;                    
          }
        }
      }
      return true;
    }

    /**
     * @brief Input thread body (100 Hz).
     *
     * 1. Calls input_interface_->update() to poll keyboard / gamepad / network.
     * 2. Forwards raw gamepad data from LowState to the active interface.
     * 3. Calls input_interface_->handle_input() to translate flags into
     *    system-state changes (motion switching, planner commands, etc.).
     * 4. Optionally records / plays back input state for offline replay.
     */
    void Input() {
      if (operator_state.stop) { return; }
      
      // Update input interface (poll for new data)
      input_interface_->update();
      
      // Handle input using the interface - input handler updates everything directly
      const std::shared_ptr<const LowState_> low_state_data = low_state_buffer_.GetDataWithTime().data;
      std::array<double, 4> current_quat = {0.0, 0.0, 0.0, 1.0};
      if(low_state_data) {
        current_quat = float_to_double<4>(low_state_data->imu_state().quaternion());
      }
      
      // Update gamepad data if using gamepad interface (or manager)
      if (auto gamepad = dynamic_cast<unitree::common::Gamepad*>(input_interface_.get())) {
        if(low_state_data) {
          memcpy(gamepad->gamepad_data.buff, &low_state_data->wireless_remote()[0], 40);
        }else{
          for(int i = 0; i < 40; i++) {
            gamepad->gamepad_data.buff[i] = 0;
          }
        }
      }
      else if (auto manager = dynamic_cast<InterfaceManager*>(input_interface_.get())) {
        if (low_state_data) {
          manager->UpdateGamepadRemoteData(&low_state_data->wireless_remote()[0], 40);
        } else {
          uint8_t zeros[40] = {0};
          manager->UpdateGamepadRemoteData(zeros, 40);
        }
      }
      else if (auto gamepad_mgr = dynamic_cast<GamepadManager*>(input_interface_.get())) {
        if (low_state_data) {
          gamepad_mgr->UpdateGamepadRemoteData(&low_state_data->wireless_remote()[0], 40);
        } else {
          uint8_t zeros[40] = {0};
          gamepad_mgr->UpdateGamepadRemoteData(zeros, 40);
        }
      }
    
     
      bool has_planner = static_cast<bool>(planner_);
      
      if (has_planner) {
        input_interface_->handle_input(motion_reader_, current_motion_, current_frame_, operator_state, 
                                      reinitialize_heading_, heading_state_buffer_, has_planner, planner_->planner_state_, movement_state_buffer_, current_motion_mutex_);
      } else {
        PlannerState planner_state{false, false};
        input_interface_->handle_input(motion_reader_, current_motion_, current_frame_, operator_state, 
                                      reinitialize_heading_, heading_state_buffer_, has_planner, planner_state, movement_state_buffer_, current_motion_mutex_);
      }

      if (playback_input_file_ && program_state_ == ProgramState::CONTROL) {
        std::string line;
        PlannerState ps{false, false};
        auto &planner_state = has_planner ? planner_->planner_state_ : ps;
        if (std::getline(*playback_input_file_, line)) {

          MovementState movement_state;
          std::istringstream iss(line);
          std::string token;
          std::getline(iss, token, ',');
          motion_reader_.current_motion_index_ = std::stoi(token);
          std::getline(iss, token, ',');
          //current_frame_ = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.play = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.start = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.stop = std::stoi(token);
          std::getline(iss, token, ',');
          planner_state.enabled = std::stoi(token);
          std::getline(iss, token, ',');
          planner_state.initialized = std::stoi(token);
          std::getline(iss, token, ',');
          movement_state.locomotion_mode = std::stoi(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[0] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[1] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[2] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[0] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[1] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[2] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_speed = std::stof(token);
          movement_state_buffer_.SetData(movement_state);

          if(!(has_planner && planner_state.enabled && planner_state.initialized)) {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
          }
        }
      }

      if (record_input_file_ && program_state_ == ProgramState::CONTROL) {
        auto movement_state_data = movement_state_buffer_.GetDataWithTime();
        PlannerState ps{false, false};
        const auto &planner_state = has_planner ? planner_->planner_state_ : ps;
        (*record_input_file_) << motion_reader_.current_motion_index_ << ",";
        (*record_input_file_) << current_frame_ << ",";
        (*record_input_file_) << operator_state.play << ",";
        (*record_input_file_) << operator_state.start << ",";
        (*record_input_file_) << operator_state.stop << ",";

        (*record_input_file_) << planner_state.enabled << ",";
        (*record_input_file_) << planner_state.initialized << ",";
        
        (*record_input_file_) << movement_state_data.data->locomotion_mode << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[0] << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[1] << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[2] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[0] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[1] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[2] << ",";
        (*record_input_file_) << movement_state_data.data->movement_speed << ",";
        (*record_input_file_) << movement_state_data.data->height << ",";
        (*record_input_file_) << std::endl;
      }

    }

    /**
     * @brief Planner thread body (10 Hz).
     *
     * When the planner is enabled:
     *  - First call: initialises the planner with the robot's current pose
     *    (base quaternion + joint positions) and runs an initial inference.
     *  - Subsequent calls: reads the latest MovementState from the
     *    movement_state_buffer_, decides whether a replan is needed (mode
     *    change, direction change, or periodic timer), and if so calls
     *    UpdatePlanning() which runs TensorRT inference and produces a new
     *    30 Hz trajectory (resampled to 50 Hz by the base class).
     *
     * Replan intervals vary by locomotion type:
     *  - Running: every 0.1 s
     *  - Crawling: every 0.2 s
     *  - Boxing: every 1.0 s
     *  - Default: every 1.0 s
     *
     * The generated animation is stored in planner_->planner_motion_50hz_
     * and picked up by CurrentFrameAdvancement() in the control thread.
     */
    void Planner() {
      if (operator_state.stop) { return; }
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      
      if (planner_ && planner_->planner_state_.enabled) {
        if (ls) {
          if (!planner_->planner_state_.initialized) {
            try {
              std::cout << "Initializing planner..." << std::endl;
              planner_motion_->timesteps = 0;
              std::cout << "Setting planner_motion encode mode to " << initial_encoder_mode_ << std::endl;
              planner_motion_->SetEncodeMode(initial_encoder_mode_);
              // Get base quaternion and joint positions from robot state
              std::array<double, 4> base_quat = float_to_double<4>(ls->imu_state().quaternion());
              std::array<double, 29> joint_positions;
              for (int i = 0; i < G1_NUM_MOTOR; i++) {
                joint_positions[i] = ls->motor_state()[i].q();
              }
              // Initialize planner with robot state
              if(!planner_->Initialize(base_quat, joint_positions)) {
                throw std::runtime_error("Error when initializing planner");
              }

              std::cout << "Planner initialized successfully!" << std::endl;
              
              // Start recording session for planner motion (if enabled)
              if (enable_motion_recording_) {
                if (planner_motion_recorder_.IsActive()) {
                  planner_motion_recorder_.Finalize();
                }
                planner_motion_recorder_.StartSession("planner_motion");
              }
              
            } catch (const std::exception& e) {
              std::cout << "✗ Error initializing planner: " << e.what() << std::endl;
              std::cout << "Disabling planner to prevent further errors..." << std::endl;
              planner_->planner_state_.enabled = false;
              planner_->planner_state_.initialized = false;
              planner_motion_->timesteps = 0;
              // reset operator state and current motion and frame
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex_);
                operator_state.play = false;
                current_frame_ = 0;
                current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
              }
              // Finalize recording if initialization failed
              if (planner_motion_recorder_.IsActive()) {
                planner_motion_recorder_.Finalize();
              }
              
              return;
            }
          } 
          else {
            if (planner_motion_->timesteps == 0) {
              std::cout << "Planner() no data in planner_motion_, returning. If control loop is started, this should only happen once when planner is initialized. If control loop is not started, this should keep returning until control loop is started." << std::endl;
              return;
            }
            bool need_replan = false;

            // bool of different reason to replan
            bool time_to_replan = false;
            bool facing_direction_changed = false;
            bool height_changed = false;
            bool movement_mode_changed = false;
            bool movement_speed_changed = false;
            bool movement_direction_changed = false;
            bool under_static_motion_mode = is_static_motion_mode(static_cast<LocomotionMode>(last_movement_state_.locomotion_mode));

            // Read current movement mode from thread-safe buffer
            auto movement_state_data = movement_state_buffer_.GetDataWithTime();

            if (movement_state_data.data) {
              // bool of checking if facing direction is changed
              facing_direction_changed = movement_state_data.data->facing_direction[0] != last_movement_state_.facing_direction[0] || 
                                          movement_state_data.data->facing_direction[1] != last_movement_state_.facing_direction[1] || 
                                          movement_state_data.data->facing_direction[2] != last_movement_state_.facing_direction[2];
              // bool of checking if height is changed
              height_changed = movement_state_data.data->height != last_movement_state_.height;
              // bool of checking if movement mode is changed
              movement_mode_changed = movement_state_data.data->locomotion_mode != last_movement_state_.locomotion_mode;
              // bool of checking if movement speed is changed
              movement_speed_changed = movement_state_data.data->movement_speed != last_movement_state_.movement_speed;
              // bool of checking if movement direction is changed
              movement_direction_changed = movement_state_data.data->movement_direction[0] != last_movement_state_.movement_direction[0] || 
                                          movement_state_data.data->movement_direction[1] != last_movement_state_.movement_direction[1] || 
                                          movement_state_data.data->movement_direction[2] != last_movement_state_.movement_direction[2];
              // bool of checking if the last movement value is under the static motion mode
              under_static_motion_mode = is_static_motion_mode(static_cast<LocomotionMode>(movement_state_data.data->locomotion_mode));
            }
            bool is_running = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RUN);
            bool is_boxing = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::LEFT_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RIGHT_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RANDOM_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::LEFT_HOOK) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RIGHT_HOOK);
            bool is_crawling = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::CRAWLING);
            // increment replan interval counter
            replan_interval_counter_ += planner_dt_;
            // check if need to replan
            if(is_running) {
              if(replan_interval_counter_ >= replan_interval_running_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else if(is_crawling) {
              if(replan_interval_counter_ >= replan_interval_crawling_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else if(is_boxing) {
              if(replan_interval_counter_ >= replan_interval_boxing_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else {
              if(replan_interval_counter_ >= replan_interval_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            }
            
            if (movement_mode_changed || facing_direction_changed || height_changed) {
              need_replan = true;
            } else if (!under_static_motion_mode && (movement_speed_changed || movement_direction_changed || (time_to_replan && movement_state_data.data->movement_speed != 0))) {
              need_replan = true;
            }

            if (!need_replan) {
              // no need to replan, return
              return;
            }

            auto gather_input_start_time = std::chrono::steady_clock::now(); 
            
            int current_mode = static_cast<int>(LocomotionMode::IDLE);
            std::array<float, 3> movement_direction = {0.0f, 0.0f, 0.0f};
            std::array<float, 3> facing_direction = {1.0f, 0.0f, 0.0f};  // Default forward
            float movement_speed = -1.0f;
            float target_height = -1.0f;
            
            if (movement_state_data.data) {
              current_mode = movement_state_data.data->locomotion_mode;
              movement_direction[0] = movement_state_data.data->movement_direction[0];
              movement_direction[1] = movement_state_data.data->movement_direction[1]; 
              movement_direction[2] = movement_state_data.data->movement_direction[2];
              facing_direction[0] = movement_state_data.data->facing_direction[0];
              facing_direction[1] = movement_state_data.data->facing_direction[1];
              facing_direction[2] = movement_state_data.data->facing_direction[2];
              movement_speed = movement_state_data.data->movement_speed;
              target_height = movement_state_data.data->height;

              // Update last movement state for next iteration comparison (update here to avoid missing the update)
              last_movement_state_.locomotion_mode = movement_state_data.data->locomotion_mode;
              last_movement_state_.movement_direction = movement_state_data.data->movement_direction;
              last_movement_state_.facing_direction = movement_state_data.data->facing_direction;
              last_movement_state_.movement_speed = movement_state_data.data->movement_speed;
              last_movement_state_.height = movement_state_data.data->height;
            }
            
            try {
              // Update planning with current movement parameters

              // gotta make sure planner_motion actually has data...
              if(!planner_->UpdatePlanning(
                current_frame_,
                planner_motion_,
                current_mode,
                movement_speed,
                target_height,
                movement_direction,
                facing_direction
              )) {
                throw std::runtime_error("Error when updating planner");
              }
              
            } catch (const std::exception& e) {
              std::cout << "✗ Error during planning update: " << e.what() << std::endl;
              std::cout << "Disabling planner to prevent further errors..." << std::endl;
              planner_->planner_state_.enabled = false;
              planner_->planner_state_.initialized = false;
              planner_motion_->timesteps = 0;
              // reset operator state and current motion and frame
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex_);
                operator_state.play = false;
                current_frame_ = 0;
                current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
              }
              // Finalize recording if planning update failed
              if (planner_motion_recorder_.IsActive()) {
                planner_motion_recorder_.Finalize();
              }
              
              return;
            }
          }
        }
      } else if (planner_ && !planner_->planner_state_.enabled) {
        // since the planner is disabled, we make sure to reset initialized and timesteps
        planner_->planner_state_.initialized = false;
        
        // Finalize planner motion recording when planner is disabled
        if (planner_motion_recorder_.IsActive()) {
          planner_motion_recorder_.Finalize();
        }
        
        return;
      }
    }

    /**
     * @brief Control thread body (50 Hz) — the main real-time control loop.
     *
     * State machine:
     *  - INIT: ramps the robot to the default standing pose (InitControl).
     *  - WAIT_FOR_CONTROL: idle, waiting for operator "start" signal.
     *  - CONTROL: each tick performs the full pipeline:
     *    1. GatherRobotStateToLogger — read IMU + joints, log to StateLogger.
     *    2. GatherInputInterfaceData — snapshot VR / hand / token data.
     *    3. GatherObservations — fill the policy observation vector.
     *    4. LogPostState — append encoder token to the latest log entry.
     *    5. CreatePolicyCommand — run TensorRT policy, produce MotorCommand.
     *    6. Update Dex3 hands (max-close ratio + joint targets).
     *    7. Publish state to all output interfaces (ZMQ / ROS2).
     *    8. Handle motion recording (streamed + planner).
     *    9. CurrentFrameAdvancement — advance playback cursor, blend planner.
     *    10. Periodic timing log every 50 ticks (~1 s).
     */
    void Control() {
      if (operator_state.stop) { return; }

      switch (program_state_) {
        case ProgramState::INIT:
          if (!InitControl()) {
            std::cout << "LowState is not available, waiting for robot to be ready" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          break;

        case ProgramState::WAIT_FOR_CONTROL:
          if (operator_state.start) {
            program_state_ = ProgramState::CONTROL;
          }
          break;

        case ProgramState::CONTROL: {
          // NEW: Get data with timestamps for loop timing analysis
          auto obs_start_time = std::chrono::steady_clock::now();
          // Increment independent logging counter every control iteration
          logging_counter_++;

          if (!GatherRobotStateToLogger()) {
            std::cout << "✗ Error: Failed to gather robot state to logger in the middle of the control loop!" << std::endl;
            operator_state.stop = true;
            std::cout << "Stopping control system." << std::endl;
            return;
          }

          if (!GatherInputInterfaceData()) {
            return;
          }

          
          // Lock mutex for observation gathering and output sending to ensure consistency
          // This ensures all observations use the same snapshot of current_motion_ and current_frame_
          int current_frame_copy;
          int current_encoder_mode_copy;
          bool current_play_copy;
          std::shared_ptr<const MotionSequence> current_motion_copy = nullptr;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_frame_copy = current_frame_;
            current_motion_copy = current_motion_;
            current_encoder_mode_copy = current_motion_copy->GetEncodeMode();
            current_play_copy = operator_state.play;

            // reset saved frame for observation window before gathering observations so that we can get the latest value
            saved_frame_for_observation_window_ = 0;
            
            // Gather all observations using modular functions
            // This may update token_state_data_ with encoder output (if encoder is used)
            if (!GatherObservations()) {
              std::cout << "✗ Error: Failed to gather observations in the middle of the control loop!" << std::endl;
              std::cout << "Stopping control system." << std::endl;
              operator_state.stop = true;
              return;
            }
          } // Release lock after all observation-dependent operations

          // Log post-state data (token state) to the most recent state logger entry
          // This must be called after GatherObservations() which populates token_state_data_
          if (state_logger_) {
            std::string motion_name = current_motion_copy ? current_motion_copy->name : "";
            if (!state_logger_->LogPostState(std::span(token_state_data_), current_encoder_mode_copy, motion_name, current_play_copy)) {
              std::cerr << "[WARNING] Failed to log token state to state logger" << std::endl;
            }
          }

          auto obs_end_time = std::chrono::steady_clock::now();

          if (!CreatePolicyCommand()) {
            std::cout << "✗ Error: Failed to create policy command in the middle of the control loop!" << std::endl;
            std::cout << "Stopping control system." << std::endl;
            operator_state.stop = true;
            return;
          }
          auto motor_command_end_time = std::chrono::steady_clock::now();

          // Update Dex3 hands max close ratio from keyboard-controlled value (X/C keys)
          dex3_hands_.SetMaxCloseRatio(input_interface_->GetMaxCloseRatio());
          
          // set hand poses (use buffered data for consistency)
          dex3_hands_.setAllJointsCommand(true, left_hand_joint_buffer_);
          dex3_hands_.setAllJointsCommand(false, right_hand_joint_buffer_);
          
          // Update last hand actions for logging (use buffered data)
          for (int i = 0; i < 7; ++i) {
            last_left_hand_action[i] = left_hand_joint_buffer_[i];
            last_right_hand_action[i] = right_hand_joint_buffer_[i];
          }
          
          auto hand_joint_end_time = std::chrono::steady_clock::now();

          // Publish output data (state logger data, robot config, command/motion data) to all output interfaces
          for (auto& output_interface : output_interfaces_) {
            if (output_interface) {
              output_interface->publish(
                vr_3point_position_buffer_, vr_3point_orientation_buffer_, vr_3point_compliance_buffer_,
                left_hand_joint_buffer_, right_hand_joint_buffer_, init_ref_data_root_rot_array_,
                heading_state_buffer_, current_motion_copy, current_frame_copy
              );
            }
          }

          // Handle recording of streamed motion (if enabled)
          if (enable_motion_recording_) {
            if (current_motion_copy && current_motion_copy->name == "streamed") {
              // Start recording session when streamed starts at frame 0
              if (current_frame_copy == 0 && !zmq_motion_recorder_.IsActive()) {
                zmq_motion_recorder_.StartSession("streamed");
              }
              
              // Write current frame if recording is active
              if (zmq_motion_recorder_.IsActive() && operator_state.play) {
                zmq_motion_recorder_.WriteFrame(current_motion_copy, current_frame_copy);
              }
            } else {
              // Finalize recording when switching away from streamed
              if (zmq_motion_recorder_.IsActive()) {
                zmq_motion_recorder_.Finalize();
              }
            }
            
            // Handle recording of planner_motion (closed-loop result)
            if (current_motion_copy && current_motion_copy == planner_motion_) {
              // Write current frame if recording is active
              if (planner_motion_recorder_.IsActive() && operator_state.play) {
                planner_motion_recorder_.WriteFrame(current_motion_copy, current_frame_copy);
              }
            }
          }

          // Write target motion to file
          if (target_motion_file_) {
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][0] << ",";
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][1] << ",";
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][2] << ",";

            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][0] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][1] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][2] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][3] << ",";

            for(int i = 0; i < 29; i++) {
              (*target_motion_file_) << current_motion_copy->JointPositions(current_frame_copy)[isaaclab_to_mujoco[i]] << ",";
            }

            (*target_motion_file_) << std::endl;
          }

          if(policy_input_file_)
          {
            // dump neural network inputs to a log file:
            for(auto d : obs_buffer_)
            {
              (*policy_input_file_) << d << ",";
            }
            (*policy_input_file_) << std::endl;
          }

          if (!CurrentFrameAdvancement()) {
            std::cout << "✗ Error: Failed to advance current frame in the middle of the control loop!" << std::endl;
            std::cout << "Stopping control system." << std::endl;
            operator_state.stop = true;
            return;
          }

          if (logging_counter_ % 50 == 0) {
            auto control_loop_end_time = std::chrono::steady_clock::now();
            auto obs_duration = std::chrono::duration_cast<std::chrono::microseconds>(obs_end_time - obs_start_time);
            auto policy_duration = std::chrono::duration_cast<std::chrono::microseconds>(motor_command_end_time - obs_end_time);
            auto motor_command_duration = std::chrono::duration_cast<std::chrono::microseconds>(motor_command_end_time - obs_start_time);
            auto post_hand_joint_duration = std::chrono::duration_cast<std::chrono::microseconds>(control_loop_end_time - hand_joint_end_time);
            
            std::cout << "Loop timing - LowState age: " << used_low_state_data_.GetAgeMs() << "ms"
                      << ", Streaming data mean delay: " << streaming_data_delay_rolling_stats_.mean() << "ms"
                      << ", Streaming data std delay: " << streaming_data_delay_rolling_stats_.stddev() << "ms"
                      << ", IMU age: " << used_imu_torso_data_.GetAgeMs() << "ms"
                      << ", Obs: " << obs_duration.count() << "us"
                      << ", Policy: " << policy_duration.count() << "us"
                      << ", Obs 2 Motor Command: " << motor_command_duration.count() << "us"
                      << ", Post processing: " << post_hand_joint_duration.count() << "us";
            
            // Add planner timing if planner is enabled and initialized
            if (planner_ && planner_->planner_state_.enabled && planner_->planner_state_.initialized) {
              std::cout << ", Planner - Gather Input: " << planner_->last_timing_.gather_input_duration.count() << "us"
                        << ", Model: " << planner_->last_timing_.model_duration.count() << "us"
                        << ", Convert50Hz: " << planner_->last_timing_.extract_duration.count() << "us"
                        << ", Total: " << planner_->last_timing_.total_duration.count() << "us";
            }
            
            // Only print compliance levels if the policy observes them
            if (has_vr_3point_compliance_obs_) {
              std::cout << " | Compliance [L,R,H]: [" 
                        << vr_3point_compliance_buffer_[0] << ", "
                        << vr_3point_compliance_buffer_[1] << ", "
                        << vr_3point_compliance_buffer_[2] << "]";
            }
            
            // Print hand max close ratio (keyboard-controlled via X/C keys)
            std::cout << " | HandCloseRatio: " << dex3_hands_.GetMaxCloseRatio();
            
            std::cout << std::endl;
          }
          break;
        }
      }
    }
};

/**
 * @brief Entry point: parse CLI arguments and run the G1 deployment application.
 *
 * Required positional arguments:
 *   1. network_interface – DDS network interface (e.g. "eth0")
 *   2. policy_file       – Path to the ONNX policy model
 *   3. motion_data_path  – Directory of pre-loaded reference motions
 *
 * All other arguments are optional flags (see --help for full list).
 * The main loop sleeps until the operator issues a stop signal or ROS2 shuts down.
 */
int main(int argc, char const* argv[]) {
  std::cout << "[DEBUG] Program starting..." << std::endl;
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <network_interface> <policy_file> <motion_data_path> [OPTIONS]"
              << std::endl;
    std::cout << "  network_interface: network interface for DDS communication" << std::endl;
    std::cout << "  policy_file: path to ONNX policy file" << std::endl;
    std::cout << "  motion_data_path: path to motion data directory (e.g., reference/bones_072925_test/)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --planner-file <path>: specify planner file (optional)" << std::endl;
    std::cout << "  --input-type <keyboard|gamepad|gamepad_manager|manager|zmq|zmq_manager";
#if HAS_ROS2
    std::cout << "|ros2";
#endif
    std::cout << ">: input interface type (default: keyboard)" << std::endl;
    std::cout << "  --output-type <zmq|all";
#if HAS_ROS2
    std::cout << "|ros2";
#endif
    std::cout << ">: output interface type (default: zmq, 'all' creates all available)" << std::endl;
    std::cout << "  --target-motion-logfile <path>: write target motion to a csv file if provided" << std::endl;
    std::cout << "  --planner-motion-logfile <path>: write planner motion to a csv file if provided" << std::endl;
    std::cout << "  --policy-input-logfile <path>: write policy input tensors to a csv file if provided" << std::endl;
    std::cout << "  --disable-crc-check: disable CRC validation for MuJoCo simulation" << std::endl;
    std::cout << "  --obs-config <path>: specify observation configuration YAML file" << std::endl;
    std::cout << "  --encoder-file <path>: specify encoder ONNX file (optional)" << std::endl;
    std::cout << "  --planner-precision <16|32>: specify precision to run the planner model at (default: 16)" << std::endl;
    std::cout << "  --policy-precision <16|32>: specify precision to run the policy model at (default: 32)" << std::endl;
    std::cout << "  --zmq-host <host>: ZMQ server host (default: localhost)" << std::endl;
    std::cout << "  --zmq-port <port>: ZMQ server port (default: 5556)" << std::endl;
    std::cout << "  --zmq-topic <topic>: ZMQ topic/prefix (default: pose)" << std::endl;
    std::cout << "  --zmq-conflate: enable ZMQ CONFLATE (default: disabled)" << std::endl;
    std::cout << "  --zmq-verbose: enable ZMQ subscriber verbose logs" << std::endl;
    std::cout << "  --zmq-out-port <port>: ZMQ port for output (default: 5557)" << std::endl;
    std::cout << "  --zmq-out-topic <topic>: ZMQ topic/prefix for output (default: g1_debug)" << std::endl;
    std::cout << "  --logs-dir <path>: optional logs output base directory (default: logs/<timestamp>/)" << std::endl;
    std::cout << "  --enable-csv-logs: enable writing CSV logs (default: OFF)" << std::endl;
    std::cout << "  --enable-motion-recording: enable motion recording for ZMQ/planner (default: OFF)" << std::endl;
    std::cout << "  --set-compliance <value>: set initial VR 3-point compliance (0.01=rigid, 0.5=compliant; default: [0.5, 0.5, 0.0])" << std::endl;
    std::cout << "                                 Can specify 1 value (both hands) or 3 values (left_wrist,right_wrist,head)" << std::endl;
    std::cout << "                                 Keyboard controls: g/h = left hand +/- 0.1, b/v = right hand +/- 0.1" << std::endl;
    std::cout << "  --max-close-ratio <value>: set initial hand max close ratio (0.2-1.0; default: 1.0 = full closure)" << std::endl;
    std::cout << "                             0.2 = limited (80% open), 1.0 = full closure allowed" << std::endl;
    std::cout << "                             Keyboard controls: x/c = +/- 0.1 (always available)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --planner-file policy/planner.onnx --obs-config policy/single_frame/observation_config.yaml --disable-crc-check" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/token/model.onnx reference/bones_072925_test/ --obs-config policy/token/observation_config.yaml --encoder-file policy/token/encoder.onnx" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type gamepad --planner-file policy/planner.onnx" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type gamepad_manager --planner-file policy/planner.onnx --zmq-host localhost --zmq-port 5556" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type zmq --zmq-host 192.168.1.2 --zmq-port 5556 --zmq-topic pose --zmq-conflate" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type zmq_manager --planner-file policy/planner.onnx --zmq-host localhost --zmq-port 5556" << std::endl;
#if HAS_ROS2
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type ros2 --planner-file policy/planner.onnx" << std::endl;
#endif
    exit(0);
  }
  std::cout << "[DEBUG] Arguments validated..." << std::endl;
  std::string networkInterface = argv[1];
  std::string modelFile = argv[2];
  std::string motionDataPath = argv[3];
  std::string plannerFile = "";

  // Parse optional arguments
  bool disableCrcCheck = false;\
  std::string obsConfigPath = "";
  std::string encoderFile = "";
  std::string targetMotionLogfile = "";
  std::string plannerMotionLogfile = "";
  std::string policyInputLogfile = "";
  std::string recordInputFile = "";
  std::string playbackInputFile = "";
  std::string inputType = "keyboard"; // Default to keyboard
  std::string outputType = "zmq"; // Default to zmq
  bool plannerFp16 = false;
  bool policyFp16 = false;
  std::string logsDir = "";
  bool enableCsvLogs = false;
  std::string zmq_host = "localhost";
  int zmq_port = 5556;
  std::string zmq_topic = "pose";
  bool zmq_conflate = false;  // default off; enable with --zmq-conflate
  bool zmq_verbose = false;
  bool enableMotionRecording = false;  // default off; enable with --enable-motion-recording
  int zmq_out_port = 5557;
  std::string zmq_out_topic = "g1_debug";
  std::array<double, 3> initial_compliance = {0.5, 0.5, 0.0}; // initial compliance is 0.5 for both hands (keyboard controllable)
  double initial_max_close_ratio = 1.0; // default allows full closure, use --max-close-ratio to limit
  for (int i = 4; i < argc; i++) {
    if (std::string(argv[i]) == "--disable-crc-check") {
      disableCrcCheck = true;
      std::cout << "[INFO] CRC checking disabled for MuJoCo simulation" << std::endl;
    } else if (std::string(argv[i]) == "--obs-config") {
      if (i + 1 < argc) {
        obsConfigPath = argv[i + 1];
        std::cout << "[INFO] Using observation config: " << obsConfigPath << std::endl;
        i++; // Skip the next argument since it's the config path
      } else {
        std::cerr << "Error: --obs-config requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--encoder-file") {
      if (i + 1 < argc) {
        encoderFile = argv[i + 1];
        std::cout << "[INFO] Using encoder file: " << encoderFile << std::endl;
        i++; // Skip the next argument since it's the encoder file path
      } else {
        std::cerr << "Error: --encoder-file requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--planner-file") {
      if (i + 1 < argc) {
        plannerFile = argv[i + 1];
        std::cout << "[INFO] Using planner file: " << plannerFile << std::endl;
        i++; // Skip the next argument since it's the planner path
      } else {
        std::cerr << "Error: --planner-file requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--target-motion-logfile") {
      if (i + 1 < argc) {
        targetMotionLogfile = argv[i + 1];
        std::cout << "[INFO] Using target motion logfile: " << targetMotionLogfile << std::endl;
        i++; // Skip the next argument since it's the target motion logfile
      }
    } else if (std::string(argv[i]) == "--planner-motion-logfile") {
      if (i + 1 < argc) {
        plannerMotionLogfile = argv[i + 1];
        std::cout << "[INFO] Using planner motion logfile: " << plannerMotionLogfile << std::endl;
        i++; // Skip the next argument since it's the planner motion logfile
      }
    } else if (std::string(argv[i]) == "--policy-input-logfile") {
      if (i + 1 < argc) {
        policyInputLogfile = argv[i + 1];
        std::cout << "[INFO] Using policy input logfile: " << policyInputLogfile << std::endl;
        i++; // Skip the next argument since it's the policy input logfile
      }
    } else if (std::string(argv[i]) == "--input-type") {
      if (i + 1 < argc) {
        inputType = argv[i + 1];
        // Validate input type based on what's available
        bool valid_input = (inputType == "keyboard" || inputType == "gamepad" || inputType == "gamepad_manager" || inputType == "zmq" || inputType == "zmq_manager" || inputType == "manager");
#if HAS_ROS2
        valid_input = valid_input || (inputType == "ros2");
#endif
        if (!valid_input) {
          std::cerr << "Error: --input-type must be 'keyboard', 'gamepad', 'gamepad_manager', 'manager', 'zmq', or 'zmq_manager'";
#if HAS_ROS2
          std::cerr << ", or 'ros2'";
#endif
          std::cerr << std::endl;
          exit(1);
        }
        std::cout << "[INFO] Using input type: " << inputType << std::endl;
        i++; // Skip the next argument since it's the input type
      } else {
        std::cerr << "Error: --input-type requires a type argument (keyboard, gamepad, gamepad_manager, manager, zmq, zmq_manager";
#if HAS_ROS2
        std::cerr << ", or ros2";
#endif
        std::cerr << ")" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--output-type") {
      if (i + 1 < argc) {
        outputType = argv[i + 1];
        bool valid_output = (outputType == "zmq" || outputType == "all");
#if HAS_ROS2
        valid_output = valid_output || (outputType == "ros2");
#endif
        if (!valid_output) {
          std::cerr << "Error: --output-type must be 'zmq', 'all'";
#if HAS_ROS2
          std::cerr << ", or 'ros2'";
#endif
          std::cerr << std::endl;
          exit(1);
        }
        std::cout << "[INFO] Using output type: " << outputType << std::endl;
        i++; // Skip the next argument since it's the output type
      } else {
        std::cerr << "Error: --output-type requires a type argument (zmq, all";
#if HAS_ROS2
        std::cerr << ", or ros2";
#endif
        std::cerr << ")" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--zmq-out-port") {
      if (i + 1 < argc) { zmq_out_port = std::stoi(argv[i + 1]); i++; }
    } else if (std::string(argv[i]) == "--zmq-out-topic") {
      if (i + 1 < argc) { zmq_out_topic = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--record-input-file") {
      if (i + 1 < argc) {
        recordInputFile = argv[i + 1];
        std::cout << "[INFO] Recording input to logfile: " << recordInputFile << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--playback-input-file") {
      if (i + 1 < argc) {
        playbackInputFile = argv[i + 1];
        std::cout << "[INFO] Playing input from logfile: " << playbackInputFile << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--planner-precision") {
      if (i + 1 < argc) {
        if(argv[i+1] == std::string("16"))
        {
          plannerFp16 = true;
        }
        else if(argv[i+1] == std::string("32"))
        {
          plannerFp16 = false;
        }
        else{
          throw std::runtime_error("invalid --planner-precision (should be 16 or 32): " + std::string(argv[i+1]));
        }
        std::cout << "[INFO] Planner precision: " << argv[i+1] << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--policy-precision") {
      std::cerr << "--policy-precision" << std::endl;
      if (i + 1 < argc) {
        if(argv[i+1] == std::string("16"))
        {
          policyFp16 = true;
        }
        else if(argv[i+1] == std::string("32"))
        {
          policyFp16 = false;
        }
        else{
          throw std::runtime_error("invalid --policy-precision (should be 16 or 32): " + std::string(argv[i+1]));
        }
        std::cout << "[INFO] Policy precision: " << argv[i+1] << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
      else{
        std::cerr << "old and weak" << std::endl;
      }
    } else if (std::string(argv[i]) == "--enable-csv-logs") {
      enableCsvLogs = true;
      std::cout << "[INFO] CSV logging enabled" << std::endl;
    } else if (std::string(argv[i]) == "--logs-dir") {
      if (i + 1 < argc) {
        logsDir = argv[i + 1];
        std::cout << "[INFO] Using logs directory: " << logsDir << std::endl;
        i++; // Skip next arg (the path)
      } else {
        std::cerr << "Error: --logs-dir requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--zmq-host") {
      if (i + 1 < argc) { zmq_host = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--zmq-port") {
      if (i + 1 < argc) { zmq_port = std::stoi(argv[i + 1]); i++; }
    } else if (std::string(argv[i]) == "--zmq-topic") {
      if (i + 1 < argc) { zmq_topic = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--zmq-conflate") {
      zmq_conflate = true;
    } else if (std::string(argv[i]) == "--zmq-verbose") {
      zmq_verbose = true;
    } else if (std::string(argv[i]) == "--enable-motion-recording") {
      enableMotionRecording = true;
      std::cout << "[INFO] Motion recording enabled" << std::endl;
    } else if (std::string(argv[i]) == "--set-compliance") {
      if (i + 1 < argc) {
        // Parse compliance values (can be 1 or 3 values)
        std::string compliance_str = argv[i + 1];
        std::vector<double> values;
        std::stringstream ss(compliance_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
          try {
            values.push_back(std::stod(token));
          } catch (...) {
            std::cerr << "Error: Invalid compliance value: " << token << std::endl;
            exit(1);
          }
        }
        if (values.size() == 1) {
          initial_compliance = {values[0], values[0], 0.0};
          std::cout << "[INFO] Initial VR 3-point compliance set to: [" << values[0] 
                    << ", " << values[0] << ", 0.0]" << std::endl;
        } else if (values.size() == 3) {
          initial_compliance = {values[0], values[1], values[2]};
          std::cout << "[INFO] Initial VR 3-point compliance set to: ["
                    << values[0] << ", " << values[1] << ", " << values[2] << "]" << std::endl;
        } else {
          std::cerr << "Error: --set-compliance requires 1 or 3 comma-separated values" << std::endl;
          exit(1);
        }
        i++; // Skip the next argument since it's the compliance value
      } else {
        std::cerr << "Error: --set-compliance requires a value argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--max-close-ratio") {
      if (i + 1 < argc) {
        try {
          initial_max_close_ratio = std::stod(argv[i + 1]);
          // Validate range [0.2, 1.0]
          if (initial_max_close_ratio < 0.2 || initial_max_close_ratio > 1.0) {
            std::cerr << "Error: --max-close-ratio must be between 0.2 and 1.0" << std::endl;
            exit(1);
          }
          std::cout << "[INFO] Initial hand max close ratio set to: " << initial_max_close_ratio 
                    << " (0.2 = limited, 1.0 = full closure)" << std::endl;
        } catch (...) {
          std::cerr << "Error: Invalid max close ratio value: " << argv[i + 1] << std::endl;
          exit(1);
        }
        i++; // Skip the next argument since it's the ratio value
      } else {
        std::cerr << "Error: --max-close-ratio requires a value argument" << std::endl;
        exit(1);
      }
    }
  }

  std::cout << "[DEBUG] Creating G1Deploy object..." << std::endl;
  G1Deploy custom(
    networkInterface,
    modelFile,
    motionDataPath,
    disableCrcCheck,
    obsConfigPath,
    encoderFile,
    plannerFile,
    targetMotionLogfile,
    plannerMotionLogfile,
    policyInputLogfile,
    inputType,
    outputType,
    recordInputFile,
    playbackInputFile,
    plannerFp16,
    policyFp16,
    logsDir,
    enableCsvLogs,
    zmq_host,
    zmq_port,
    zmq_topic,
    zmq_conflate,
    zmq_verbose,
    zmq_out_port,
    zmq_out_topic,
    enableMotionRecording,
    initial_compliance,
    initial_max_close_ratio
  );
  std::cout << "[DEBUG] G1Deploy object created successfully!" << std::endl;
  
  // Main application loop - check both operator_state.stop and ROS2 status if using ROS2
#if HAS_ROS2
  if (inputType == "ros2") {
    while (!custom.operator_state.stop && rclcpp::ok()) { 
      sleep(0.02); 
    }
    if (!rclcpp::ok()) {
      std::cout << "[INFO] ROS2 shutdown detected (Ctrl+C)" << std::endl;
    }
  } else {
    while (!custom.operator_state.stop) { sleep(0.02); }
  }
#else
  while (!custom.operator_state.stop) { sleep(0.02); }
#endif
  
  std::cout << "[DEBUG] Stopping G1Deploy..." << std::endl;
  custom.Stop();
  std::cout << "[DEBUG] Waiting for cleanup..." << std::endl;
  sleep(0.5);
  std::cout << "[DEBUG] Program exiting normally..." << std::endl;
  return 0;
}
