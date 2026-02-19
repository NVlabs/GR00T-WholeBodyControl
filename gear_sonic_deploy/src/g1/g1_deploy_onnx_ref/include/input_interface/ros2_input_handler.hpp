/**
 * @file ros2_input_handler.hpp
 * @brief ROS 2 input handler for receiving teleop commands over DDS.
 *
 * ROS2InputHandler creates a lightweight ROS 2 node and subscribes to a single
 * topic (`ControlPolicy/upper_body_pose`) that carries msgpack-serialised
 * ControlGoalMsg payloads inside a `std_msgs/ByteMultiArray`.
 *
 * ## Data Flow
 *
 *   Python teleop script  ──(msgpack/ROS2)──►  ROS2InputHandler
 *       ControlGoalMsg                           ↓
 *                                           update()  (spin + buffer → local state)
 *                                           handle_input()  (local state → system state)
 *
 * ## Operational Requirements
 *
 * - A locomotion **planner must be loaded** – ROS2 mode always operates through
 *   the planner (no reference-motion playback).
 * - VR 3-point tracking is always enabled; the handler populates position and
 *   orientation buffers from the received wrist / head data.
 * - Supports two IK modes controlled at construction time:
 *   - `use_ik_mode = true`  – uses IK-processed transformation matrices
 *     (left_wrist_after_ik, right_wrist_after_ik, head_after_ik) with
 *     configurable wrist offsets.
 *   - `use_ik_mode = false` – uses raw wrist matrices (left_wrist, right_wrist).
 *
 * ## Locomotion Modes
 *
 *   base_height_command | Mode
 *   --------------------|------
 *   0.72 – 0.88         | WALK (or SLOW_WALK depending on locomotion_mode flag)
 *   0.50 – 0.72         | SQUAT (static)
 *   0.10 – 0.50         | KNEEL (static)
 *
 * ## Edge-Triggered Commands
 *
 * `toggle_policy_action` is accumulated with OR logic in the callback so that
 * a transient toggle pulse is never lost between update() cycles.
 *
 * ## Thread Safety
 *
 * The subscriber callback runs on the ROS 2 executor thread.  Data is copied
 * into `control_goal_buffer_` under `control_goal_mutex_` and consumed by
 * update() on the main thread.
 */

#ifndef ROS2_INPUT_HANDLER_HPP
#define ROS2_INPUT_HANDLER_HPP

#if HAS_ROS2
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/exceptions/exceptions.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>  // For msgpack-serialized messages
#include <array>
#include <memory>
#include <atomic>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>
#include <map>
#include <vector>
#include <variant>
#include <cstdlib>

#include <termios.h>
#include <fcntl.h>
#include <unistd.h>

#include <msgpack.hpp>

#include "input_interface.hpp"
#include "../math_utils.hpp"
#include "../policy_parameters.hpp"  // For isaaclab_to_mujoco and default_angles

/**
 * @brief Deserialized control-goal message received from the Python teleop script.
 *
 * Fields are populated by parse_msgpack_control_goal() and stored in the
 * receiving buffer for consumption by update().
 */
struct ControlGoalMsg {
    /// Navigation velocity command [lin_vel_x, lin_vel_y, ang_vel_z] (m/s, m/s, rad/s).
    std::array<double, 3> navigate_cmd = {0.0f, 0.0f, 0.0f};
    /// Wrist pose in [x,y,z, qw,qx,qy,qz] × 2 (left then right).
    std::array<double, 14> wrist_pose = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    
    // IK-processed wrist poses (4×4 transformation matrices, row-major flattened)
    std::array<double, 16> left_wrist_after_ik = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};   ///< Left wrist after IK.
    std::array<double, 16> right_wrist_after_ik = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};  ///< Right wrist after IK.
    std::array<double, 16> head_after_ik = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};         ///< Head after IK.
    bool has_ik_data = false;  ///< True if the IK-processed matrices are present.
    
    // Non-IK raw wrist matrices (4×4 transformation matrices, row-major flattened)
    std::array<double, 16> left_wrist = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};   ///< Raw left wrist transform.
    std::array<double, 16> right_wrist = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};  ///< Raw right wrist transform.
    bool has_wrist_matrices = false;  ///< True if the raw wrist matrices are present.
    
    double base_height_command = 0.78;        ///< Desired base height (metres, valid range 0.1–0.88).
    bool toggle_policy_action = false;        ///< Edge-triggered toggle: maps to start/stop control.
    int locomotion_mode = 0;                  ///< 0 = slow walk (custom speed), 1 = fast walk (default speed).
    
    /// Dex3 hand joint positions (7 DOF per hand).
    std::array<double, 7> left_hand_joint = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 7> right_hand_joint = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool has_hand_joints = false;  ///< True if hand joint data is present.
    
    double ros_timestamp = 0.0;  ///< ROS time in seconds (for synchronisation with other components).
    bool valid = false;           ///< True once the message has been successfully parsed.
};

/**
 * @class ROS2InputHandler
 * @brief InputInterface driven by ROS 2 DDS messages (msgpack-serialised ControlGoalMsg).
 *
 * Architecture:
 *  1. **ROS 2 callback** (`control_goal_callback`) receives ByteMultiArray messages,
 *     deserialises them via msgpack, and stores the result in `control_goal_buffer_`
 *     under `control_goal_mutex_`.
 *  2. **update()** spins the ROS 2 node, reads the buffer, and converts the data
 *     into local state (VR buffers, navigate_cmd, control flags).
 *  3. **handle_input()** translates local state into system-state changes
 *     (planner commands, operator start/stop, movement buffer updates).
 *
 * A 1-second timeout (`CONTROL_GOAL_TIMEOUT`) resets the handler if messages
 * stop arriving, and any ROS 2 errors trigger an immediate emergency stop.
 */
class ROS2InputHandler : public InputInterface {
public:
    // ========================================
    // DEBUG CONTROL FLAG
    // ========================================
    static constexpr bool DEBUG_LOGGING = true;  // Set to false to disable debug logs

    // Constructor - initializes ROS2 node and subscribers
    explicit ROS2InputHandler(bool use_ik_mode = true, const std::string& node_name = "g1_input_handler") 
        : InputInterface() {
        // Set terminal to non-blocking mode for keyboard input (emergency stop)
        tcgetattr(STDIN_FILENO, &old_termios_);
        struct termios new_termios = old_termios_;
        new_termios.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
        
        // Initialize ROS2 - this is the only ROS2 component in the system
        if (!rclcpp::ok()) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 DEBUG] Initializing ROS2" << std::endl;
            }
            rclcpp::init(0, nullptr);  // Initialize with no command line arguments
        }
        
        try {
            // Initialize ROS2 node
            node_ = rclcpp::Node::make_shared(node_name);
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 DEBUG] ROS2InputHandler node '" << node_name << "' initialized" << std::endl;
            }
            
            // Setup subscriber for control goal topic
            setup_subscribers();
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 DEBUG] Subscribed to topic: ControlPolicy/upper_body_pose" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ROS2 ERROR] Failed to initialize ROS2InputHandler: " << e.what() << std::endl;
            throw;
        }
        type_ = InputType::ROS2;
        has_vr_3point_control_ = true;
        use_ik_mode_ = use_ik_mode;
    }

    // Destructor
    ~ROS2InputHandler() {
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ROS2 DEBUG] ROS2InputHandler destructor called" << std::endl;
        }
        
        // Restore terminal before ROS2 cleanup
        tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
        
        // CRITICAL: Proper shutdown order to avoid EntityDelegate assertion
        try {
            // Step 1: Stop spinning to prevent new callbacks
            if (node_) {
                // Ensure no more spin operations happen
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Step 2: Reset subscribers BEFORE resetting node (proper DDS entity order)
            if (control_goal_sub_) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 DEBUG] Resetting control goal subscriber" << std::endl;
                }
                control_goal_sub_.reset();
            }
            
            // Step 3: Allow DDS cleanup time (critical for preventing assertion)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Step 4: Reset node AFTER subscribers are cleaned up
            if (node_) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 DEBUG] Resetting ROS2 node" << std::endl;
                }
                node_.reset();
            }
            
            // Step 5: Only shutdown if we're the last ROS2 component
            // Note: Be careful about calling shutdown() if other ROS2 components exist
            if (rclcpp::ok()) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 DEBUG] Shutting down ROS2 context" << std::endl;
                }
                rclcpp::shutdown();
            }
            
        } catch (const std::exception& e) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 ERROR] Exception during cleanup: " << e.what() << std::endl;
            }
            // Don't rethrow in destructor
        } catch (...) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 ERROR] Unknown exception during cleanup" << std::endl;
            }
        }
    }

    // Flag to trigger safety reset in handle_input
    bool trigger_safety_reset = false;
    
    // Flag to trigger emergency stop (set internally on ROS2 errors/timeout)
    bool emergency_stop_ = false;

    // Override the update function from InputInterface
    // Reads from control goal buffer (updated by callback) and updates local state
    // 
    // EDGE-TRIGGERED BEHAVIOR:
    // Toggle commands like toggle_policy_action are accumulated in the callback using OR logic.
    // Once a command arrives as 'true', it stays 'true' in the buffer until processed here.
    // After reading, these commands are cleared from the buffer to prevent re-execution.
    void update() override {
        // Reset emergency stop flag each frame
        emergency_stop_ = false;
        
        // Check for safety reset trigger from manager
        if (CheckAndClearSafetyReset()) {
            trigger_safety_reset = true;
            std::cout << "[ROS2InputHandler] Safety reset triggered: will disable planner and return to reference motion" << std::endl;
        }

        // Check if ROS2 is still healthy
        if (!rclcpp::ok()) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 ERROR] ROS2 context is not OK - triggering emergency stop" << std::endl;
            }
            emergency_stop_ = true;
            return;  // Skip further processing
        }

        // First, spin ROS2 node to process any pending messages and trigger callbacks
        if (node_) {
            try {
                rclcpp::spin_some(node_);
            } catch (const rclcpp::exceptions::RCLError& e) {
                std::cerr << "[ROS2 ERROR] RCL error during spin: " << e.what() << " - triggering emergency stop" << std::endl;
                emergency_stop_ = true;
                return;  // Skip further processing
            } catch (const std::exception& e) {
                std::cerr << "[ROS2 ERROR] Failed to spin node: " << e.what() << " - triggering emergency stop" << std::endl;
                emergency_stop_ = true;
                return;  // Skip further processing
            }
        }
        
        // Reset input flags each frame
        start_control_ = false;
        stop_control_ = false;

        // Read keyboard input for emergency stop ('O'/'o' key)
        // This works in standalone mode; when managed by InterfaceManager, it's also handled there
        char ch;
        while (ReadStdinChar(ch)) {
            switch (ch) {
                case 'o':
                case 'O':
                    stop_control_ = true;
                    std::cout << "[ROS2] Emergency stop triggered (O/o key pressed)" << std::endl;
                    break;
                // Ignore other keys - ROS2 is message-driven
            }
        }
        
        // Check for control goal timeout (using steady_clock for monotonic timing)
        if (received_control_goal_.load()) {
            int64_t current_time_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            int64_t last_msg_time_ns = last_control_goal_time_ns_.load();
            double time_since_last_msg = (current_time_ns - last_msg_time_ns) / 1e9;  // Convert ns to seconds
            if (time_since_last_msg > CONTROL_GOAL_TIMEOUT) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 WARNING] Control goal timeout (" << time_since_last_msg 
                              << "s since last message). Resetting flags." << std::endl;
                }
                reset_data_flags();  // Reset both flag and timestamp consistently
                use_teleop_navigate_cmd_ = false;
                planner_facing_angle_ = 0.0; // Reset facing angle
            }
        }

        // Read from control goal buffer (teleop commands from Python) - thread-safe
        if (received_control_goal_.load()) {
            std::lock_guard<std::mutex> lock(control_goal_mutex_);
            // Update navigate_cmd and base_height_command from control goal
            navigate_cmd_from_teleop_ = control_goal_buffer_.navigate_cmd;
            base_height_command_ = control_goal_buffer_.base_height_command;
            use_teleop_navigate_cmd_ = true;
            
            // Handle toggle_policy_action (edge-triggered toggle between start/stop)
            if (control_goal_buffer_.toggle_policy_action) {
                // Toggle the control state
                control_is_active_ = !control_is_active_;
                
                if (control_is_active_) {
                    start_control_ = true;
                    if constexpr (DEBUG_LOGGING) {
                        std::cout << "[ROS2 DEBUG] toggle_policy_action: START control" << std::endl;
                    }
                } else {
                    stop_control_ = true;
                    if constexpr (DEBUG_LOGGING) {
                        std::cout << "[ROS2 DEBUG] toggle_policy_action: STOP control" << std::endl;
                    }
                }
                
                // Clear the trigger after reading (edge-triggered behavior)
                control_goal_buffer_.toggle_policy_action = false;
            }
            
            // Handle locomotion_mode (direct state: 0 = slow walk, 1 = fast walk)
            locomotion_mode_is_fast_ = (control_goal_buffer_.locomotion_mode == 1);
            
            if constexpr (DEBUG_LOGGING) {
                static int prev_locomotion_mode = -1;
                if (prev_locomotion_mode != control_goal_buffer_.locomotion_mode) {
                    if (locomotion_mode_is_fast_) {
                        std::cout << "[ROS2 DEBUG] locomotion_mode: FAST WALK (default speed, mode 2)" << std::endl;
                    } else {
                        std::cout << "[ROS2 DEBUG] locomotion_mode: SLOW WALK (custom speed, mode 1)" << std::endl;
                    }
                    prev_locomotion_mode = control_goal_buffer_.locomotion_mode;
                }
            }
            
            // Update hand poses from teleop
            if (control_goal_buffer_.has_hand_joints) {
                left_hand_joint_.SetData(control_goal_buffer_.left_hand_joint);
                right_hand_joint_.SetData(control_goal_buffer_.right_hand_joint);
                has_hand_joints_ = true;
            }
            
            // Update VR 3-point control data based on use_ik_mode_ flag
            // Build arrays first, then call SetData() on buffers
            std::array<double, 9> vr_position;
            std::array<double, 12> vr_orientation;
            
            if (use_ik_mode_ && control_goal_buffer_.has_ik_data) {
                // IK mode: Use IK-processed transformation matrices
                // Offsets from local_vr_tracking_bm.py (lines 39-41)
                // NOTE the left and right hand offsets are opposite in sign for the y-axis
                constexpr std::array<double, 3> LEFT_HAND_OFFSET = {0.18, -0.025, 0.0};
                constexpr std::array<double, 3> RIGHT_HAND_OFFSET = {0.18, +0.025, 0.0};
                constexpr std::array<double, 3> HEAD_OFFSET = {0.0, 0.0, 0.35};
                
                // Extract left wrist
                auto left_rot = extract_rotation_from_transform(control_goal_buffer_.left_wrist_after_ik);
                auto left_pos = extract_position_from_transform(control_goal_buffer_.left_wrist_after_ik);
                auto left_offset_rotated = apply_rotation_to_offset(left_rot, LEFT_HAND_OFFSET);
                vr_position[0] = left_pos[0] + left_offset_rotated[0];
                vr_position[1] = left_pos[1] + left_offset_rotated[1];
                vr_position[2] = left_pos[2] + left_offset_rotated[2];
                
                // Extract right wrist
                auto right_rot = extract_rotation_from_transform(control_goal_buffer_.right_wrist_after_ik);
                auto right_pos = extract_position_from_transform(control_goal_buffer_.right_wrist_after_ik);
                auto right_offset_rotated = apply_rotation_to_offset(right_rot, RIGHT_HAND_OFFSET);
                vr_position[3] = right_pos[0] + right_offset_rotated[0];
                vr_position[4] = right_pos[1] + right_offset_rotated[1];
                vr_position[5] = right_pos[2] + right_offset_rotated[2];
                
                // Extract head
                auto head_rot = extract_rotation_from_transform(control_goal_buffer_.head_after_ik);
                auto head_pos = extract_position_from_transform(control_goal_buffer_.head_after_ik);
                auto head_offset_rotated = apply_rotation_to_offset(head_rot, HEAD_OFFSET);
                vr_position[6] = head_pos[0] + head_offset_rotated[0];
                vr_position[7] = head_pos[1] + head_offset_rotated[1];
                vr_position[8] = head_pos[2] + head_offset_rotated[2];
                
                // Convert rotation matrices to quaternions using math_utils.hpp
                auto left_quat = rotation_matrix_to_quat_d(left_rot);
                auto right_quat = rotation_matrix_to_quat_d(right_rot);
                auto head_quat = rotation_matrix_to_quat_d(head_rot);
                
                // Store quaternions (w, x, y, z format)
                vr_orientation[0] = left_quat[0];   // left qw
                vr_orientation[1] = left_quat[1];   // left qx
                vr_orientation[2] = left_quat[2];   // left qy
                vr_orientation[3] = left_quat[3];   // left qz
                vr_orientation[4] = right_quat[0];  // right qw
                vr_orientation[5] = right_quat[1];  // right qx
                vr_orientation[6] = right_quat[2];  // right qy
                vr_orientation[7] = right_quat[3];  // right qz
                vr_orientation[8] = head_quat[0];   // head qw
                vr_orientation[9] = head_quat[1];   // head qx
                vr_orientation[10] = head_quat[2];  // head qy
                vr_orientation[11] = head_quat[3];  // head qz
                
                // Update buffers
                vr_3point_position_.SetData(vr_position);
                vr_3point_orientation_.SetData(vr_orientation);
            } else if (!use_ik_mode_ && control_goal_buffer_.has_wrist_matrices) {
                // Non-IK mode with wrist matrices: Use left_wrist and right_wrist matrices
                // Extract left wrist (matches Python: left_wrist_matrix[:3, 3:] for position)
                auto left_rot = extract_rotation_from_transform(control_goal_buffer_.left_wrist);
                auto left_pos = extract_position_from_transform(control_goal_buffer_.left_wrist);
                vr_position[0] = left_pos[0];
                vr_position[1] = left_pos[1];
                vr_position[2] = left_pos[2];
                
                // Extract right wrist
                auto right_rot = extract_rotation_from_transform(control_goal_buffer_.right_wrist);
                auto right_pos = extract_position_from_transform(control_goal_buffer_.right_wrist);
                vr_position[3] = right_pos[0];
                vr_position[4] = right_pos[1];
                vr_position[5] = right_pos[2];
                // Head position uses defaults
                vr_position[6] = 0.0241;
                vr_position[7] = -0.0081;
                vr_position[8] = 0.4028;
                
                // Convert rotation matrices to quaternions using math_utils.hpp
                auto left_quat = rotation_matrix_to_quat_d(left_rot);
                auto right_quat = rotation_matrix_to_quat_d(right_rot);
                
                // Store quaternions (w, x, y, z format)
                vr_orientation[0] = left_quat[0];   // left qw
                vr_orientation[1] = left_quat[1];   // left qx
                vr_orientation[2] = left_quat[2];   // left qy
                vr_orientation[3] = left_quat[3];   // left qz
                vr_orientation[4] = right_quat[0];  // right qw
                vr_orientation[5] = right_quat[1];  // right qx
                vr_orientation[6] = right_quat[2];  // right qy
                vr_orientation[7] = right_quat[3];  // right qz
                // Head orientation uses defaults
                vr_orientation[8] = 0.9991;
                vr_orientation[9] = 0.011;
                vr_orientation[10] = 0.0402;
                vr_orientation[11] = -0.0002;
                
                // Update buffers
                vr_3point_position_.SetData(vr_position);
                vr_3point_orientation_.SetData(vr_orientation);
            } else {
                // Fallback: Use standard wrist_pose format (14 doubles)
                vr_position[0] = control_goal_buffer_.wrist_pose[0];  // left wrist x
                vr_position[1] = control_goal_buffer_.wrist_pose[1];  // left wrist y
                vr_position[2] = control_goal_buffer_.wrist_pose[2];  // left wrist z
                vr_position[3] = control_goal_buffer_.wrist_pose[7];  // right wrist x
                vr_position[4] = control_goal_buffer_.wrist_pose[8];  // right wrist y
                vr_position[5] = control_goal_buffer_.wrist_pose[9];  // right wrist z
                // Head position uses defaults
                vr_position[6] = 0.0241;
                vr_position[7] = -0.0081;
                vr_position[8] = 0.4028;
                
                // Update VR 3-point orientation data (wrist quaternions)
                vr_orientation[0] = control_goal_buffer_.wrist_pose[3];   // left wrist qw
                vr_orientation[1] = control_goal_buffer_.wrist_pose[4];   // left wrist qx
                vr_orientation[2] = control_goal_buffer_.wrist_pose[5];   // left wrist qy
                vr_orientation[3] = control_goal_buffer_.wrist_pose[6];   // left wrist qz
                vr_orientation[4] = control_goal_buffer_.wrist_pose[10];  // right wrist qw
                vr_orientation[5] = control_goal_buffer_.wrist_pose[11];  // right wrist qx
                vr_orientation[6] = control_goal_buffer_.wrist_pose[12];  // right wrist qy
                vr_orientation[7] = control_goal_buffer_.wrist_pose[13];  // right wrist qz
                // Head orientation uses defaults
                vr_orientation[8] = 0.9991;
                vr_orientation[9] = 0.011;
                vr_orientation[10] = 0.0402;
                vr_orientation[11] = -0.0002;
                
                // Update buffers
                vr_3point_position_.SetData(vr_position);
                vr_3point_orientation_.SetData(vr_orientation);
            }

            if constexpr (DEBUG_LOGGING) {
                static int goal_debug_counter = 0;
                goal_debug_counter++;
                if (goal_debug_counter % 50 == 0) {  // Log every 50 calls to avoid spam
                    std::cout << "[ROS2 DEBUG] Control goal update:" << std::endl;
                    std::cout << "  Navigate cmd: [" << navigate_cmd_from_teleop_[0] << ", " 
                              << navigate_cmd_from_teleop_[1] << ", " << navigate_cmd_from_teleop_[2] << "]" << std::endl;
                    std::cout << "  Base height: " << control_goal_buffer_.base_height_command << std::endl;
                    std::cout << "  Toggle policy action: " << (control_goal_buffer_.toggle_policy_action ? "true" : "false") << std::endl;
                    
                    // Data availability flags
                    std::cout << "  Data flags: has_ik_data=" << (control_goal_buffer_.has_ik_data ? "true" : "false")
                              << ", has_wrist_matrices=" << (control_goal_buffer_.has_wrist_matrices ? "true" : "false")
                              << ", has_hand_joints=" << (has_hand_joints_ ? "true" : "false")
                              << ", use_ik_mode=" << (use_ik_mode_ ? "true" : "false") << std::endl;
                    
                    // VR 3-point positions (left wrist, right wrist, head)
                    std::cout << "  VR 3-point positions:" << std::endl;
                    std::cout << "    Left wrist:  [" << vr_position[0] << ", " << vr_position[1] << ", " << vr_position[2] << "]" << std::endl;
                    std::cout << "    Right wrist: [" << vr_position[3] << ", " << vr_position[4] << ", " << vr_position[5] << "]" << std::endl;
                    std::cout << "    Head:        [" << vr_position[6] << ", " << vr_position[7] << ", " << vr_position[8] << "]" << std::endl;
                    
                    // VR 3-point orientations (quaternions: w, x, y, z)
                    std::cout << "  VR 3-point orientations:" << std::endl;
                    std::cout << "    Left wrist:  [" << vr_orientation[0] << ", " << vr_orientation[1] << ", " 
                              << vr_orientation[2] << ", " << vr_orientation[3] << "]" << std::endl;
                    std::cout << "    Right wrist: [" << vr_orientation[4] << ", " << vr_orientation[5] << ", " 
                              << vr_orientation[6] << ", " << vr_orientation[7] << "]" << std::endl;
                    std::cout << "    Head:        [" << vr_orientation[8] << ", " << vr_orientation[9] << ", " 
                              << vr_orientation[10] << ", " << vr_orientation[11] << "]" << std::endl;
                    
                    if (has_hand_joints_) {
                        auto [has_left, left_hand] = GetHandPose(true);
                        auto [has_right, right_hand] = GetHandPose(false);
                        std::cout << "  Left hand pose: [";
                        for (size_t i = 0; i < 7; ++i) {
                            std::cout << left_hand[i];
                            if (i < 6) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                        std::cout << "  Right hand pose: [";
                        for (size_t i = 0; i < 7; ++i) {
                            std::cout << right_hand[i];
                            if (i < 6) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    }
                }
            }
        } else {
            // No control goal data available
            use_teleop_navigate_cmd_ = false;
        }
    }

    // Override the handle_input function from InputInterface
    // Uses local boolean flags (set by update()) to perform actions on system state
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
        
        // Handle emergency stop (triggered by ROS2 errors/timeout)
        if (emergency_stop_) {
            std::cout << "[ROS2 EMERGENCY STOP] Triggering emergency stop due to ROS2 failure" << std::endl;
            operator_state.stop = true;
            return;  // Skip all other processing
        }
        
        // Handle safety reset from interface manager
        // ROS2 requires planner to be enabled
        if (trigger_safety_reset) {
            trigger_safety_reset = false;
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
            }
            if (has_planner && operator_state.start) {
                control_is_active_ = true;
                if (planner_state.enabled && planner_state.initialized) {
                    // Planner is already on, keep it as is (don't touch initialized flag)
                    {
                        std::lock_guard<std::mutex> lock(current_motion_mutex);
                        if (current_motion->GetEncodeMode() >= 0) {
                            current_motion->SetEncodeMode(1);
                        }
                        operator_state.play = true;
                    }
                    auto current_facing = movement_state_buffer.GetDataWithTime().data->facing_direction;
                    planner_facing_angle_ = std::atan2(current_facing[1], current_facing[0]);
                    std::cout << "Safety reset: Planner kept enabled with current state" << std::endl;
                } else {
                    // Planner was disabled, go to first motion and set initial heading and movement state
                    // Set initial heading and movement state
                    movement_state_buffer.SetData(MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f));
                    // Set current motion and frame to reference motion (lock mutex)
                    {
                        std::lock_guard<std::mutex> lock(current_motion_mutex);
                        auto temp_motion = std::make_shared<MotionSequence>(*current_motion);
                        temp_motion->name = "temporary_motion";
                        current_motion = temp_motion;
                    }

                    // Now it is safe to enable planner
                    // Ensure planner is enabled (always required in ROS2 mode)
                    planner_state.enabled = true;
                    planner_facing_angle_ = 0.0;
                    std::cout << "[ROS2] Planner enabled" << std::endl;
                    // Wait for planner to be initialized with timeout (5 seconds)
                    auto wait_start = std::chrono::steady_clock::now();
                    constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
                    while (planner_state.enabled) {
                        {
                            std::lock_guard<std::mutex> lock(current_motion_mutex);
                            if (current_motion->name == "planner_motion") {
                            std::cout << "[ROS2] motion name is planner_motion" << std::endl;
                            break;
                            }
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        auto elapsed = std::chrono::steady_clock::now() - wait_start;
                        if (elapsed > PLANNER_INIT_TIMEOUT) {
                            std::cerr << "[ROS2 ERROR] Planner initialization timeout after 5 seconds" << std::endl;
                            operator_state.stop = true;
                            return;
                        }
                        std::cout << "[ROS2] Waiting for planner to be initialized" << std::endl;
                    }
                    // Check if planner is enabled and initialized
                    if (!planner_state.enabled || !planner_state.initialized) {
                        std::cerr << "[ROS2 ERROR] Planner failed to initialize - ROS2 mode requires planner. Stopping control." << std::endl;
                        operator_state.stop = true;
                        return;
                    }
                    // Play motion
                    {
                        std::lock_guard<std::mutex> lock(current_motion_mutex);
                        if (current_motion->GetEncodeMode() == 0) {
                            current_motion->SetEncodeMode(1);
                        }
                        operator_state.play = true;
                    }
                }
            }
        }

        // Check if planner is loaded (required for ROS2 mode)
        if (!has_planner) {
            std::cerr << "[ROS2 ERROR] Planner not loaded - ROS2 mode requires planner. Stopping control." << std::endl;
            operator_state.stop = true;
            return;
        }

        // Handle control start/stop
        if (this->stop_control_) { operator_state.stop = true; }

        // Handle control start
        if (this->start_control_) { 
            // Start control
            operator_state.start = true;
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                reinitialize_heading = true;
            }
            // Ensure planner is enabled (always required in ROS2 mode)
            if (!planner_state.enabled) {
                planner_state.enabled = true;
                planner_facing_angle_ = 0.0;
                std::cout << "[ROS2] Planner enabled" << std::endl;
            }
            // Wait for planner to be initialized with timeout (5 seconds)
            auto wait_start = std::chrono::steady_clock::now();
            constexpr auto PLANNER_INIT_TIMEOUT = std::chrono::seconds(5);
            while (planner_state.enabled) {
                {
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    if (current_motion->name == "planner_motion") {
                        std::cout << "[ROS2] motion name is planner_motion" << std::endl;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                auto elapsed = std::chrono::steady_clock::now() - wait_start;
                if (elapsed > PLANNER_INIT_TIMEOUT) {
                    std::cerr << "[ROS2 ERROR] Planner initialization timeout after 5 seconds" << std::endl;
                    operator_state.stop = true;
                    return;
                }
                std::cout << "[ROS2] Waiting for planner to be initialized" << std::endl;
            }
            // Check if planner is enabled and initialized
            if (!planner_state.enabled || !planner_state.initialized) {
                std::cerr << "[ROS2 ERROR] Planner failed to initialize - ROS2 mode requires planner. Stopping control." << std::endl;
                operator_state.stop = true;
                return;
            }
            // Play motion
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = true;
            }
        }

        if (planner_state.enabled && planner_state.initialized) {
            
            // Set final movement values from navigate_cmd
            int final_mode = static_cast<int>(LocomotionMode::IDLE);
            std::array<double, 3> final_movement = {0.0, 0.0, 0.0};
            std::array<double, 3> final_facing_direction = {1.0, 0.0, 0.0};
            double final_speed = 0.0;
            double final_height = -1.0;
            
            // Get and process base_height_command (thread-safe copy)
            double base_height = base_height_command_;
            // Clip to valid range [0.1, 0.88]
            base_height = std::clamp(base_height, 0.1, 0.88);
            
            // Convert navigate_cmd to movement direction and mode
            if (use_teleop_navigate_cmd_) {
                // navigate_cmd format: [lin_vel_x, lin_vel_y, ang_vel_z]
                // Convert to movement_direction and mode
                double lin_vel_x = navigate_cmd_from_teleop_[0];
                double lin_vel_y = navigate_cmd_from_teleop_[1];
                double ang_vel_z = navigate_cmd_from_teleop_[2];
                
                // Update facing angle based on angular velocity (similar to gamepad)
                if (std::abs(ang_vel_z) > 0.01f) {
                    // Negative sign matches gamepad behavior (positive ang_vel_z = turn left/CCW)
                    planner_facing_angle_ += ang_vel_z * 0.02;
                }
                
                // Always compute facing direction from maintained angle (not from navigation topic)
                final_facing_direction[0] = std::cos(planner_facing_angle_);
                final_facing_direction[1] = std::sin(planner_facing_angle_);
                final_facing_direction[2] = 0.0f;
                
                // Calculate movement magnitude
                double movement_mag = std::sqrt(lin_vel_x * lin_vel_x + lin_vel_y * lin_vel_y);

                double planner_moving_direction = planner_facing_angle_;
                
                // Determine locomotion mode based on base_height_command first
                if (base_height >= 0.72f) {
                    // Height 0.72-0.88: Normal walking modes
                    if (movement_mag > 0.01f) {
                        // Moving: use walk modes
                        // Compute moving direction (same as gamepad logic)
                        planner_moving_direction = std::atan2(lin_vel_y, lin_vel_x) + planner_moving_direction;
                        
                        // Bin the moving direction to 8 evenly spaced directions and get corresponding speed
                        auto [binned_angle, direction_speed] = bin_angle_to_8_directions(planner_moving_direction);
                        planner_moving_direction = binned_angle;
                        
                        // Compute normalized movement direction from binned angle
                        final_movement[0] = std::cos(planner_moving_direction);
                        final_movement[1] = std::sin(planner_moving_direction);
                        final_movement[2] = 0.0f;
                        
                        if (locomotion_mode_is_fast_) {
                            // Normal walk mode: default speed (-1)
                            final_mode = static_cast<int>(LocomotionMode::WALK);
                            final_speed = -1.0f;
                        } else {
                            // Slow walk mode: speed varies by direction (faster forward/lateral, slower backward)
                            final_mode = static_cast<int>(LocomotionMode::SLOW_WALK);
                            final_speed = direction_speed;
                        }
                    } else {
                        // No movement: idle
                        final_mode = static_cast<int>(LocomotionMode::IDLE);
                        final_movement = {0.0f, 0.0f, 0.0f};
                        final_speed = -1.0f;
                    }
                    final_height = -1.0f;  // Use default height for walking
                    
                } else if (base_height >= 0.5f) {
                    // Height 0.5-0.72: Squat mode (static pose, no movement)
                    final_mode = static_cast<int>(LocomotionMode::IDEL_SQUAT);
                    final_movement = {0.0f, 0.0f, 0.0f};
                    final_speed = -1.0f;  // Use default speed (no walking while squatting)
                    final_height = base_height;  // Pass actual height command
                    
                } else {
                    // Height 0.1-0.5: Kneel mode (static pose, no movement)
                    final_mode = static_cast<int>(LocomotionMode::IDEL_KNEEL);
                    final_movement = {0.0f, 0.0f, 0.0f};
                    final_speed = -1.0f;  // Use default speed (no walking while kneeling)
                    final_height = base_height;  // Pass actual height command
                }
            }

            // Debug: Log final computed values being sent to planner
            if constexpr (DEBUG_LOGGING) {
                static int debug_counter = 0;
                debug_counter++;
                if (debug_counter % 50 == 0) {  // Log every 50 calls to avoid spam
                    std::cout << "[ROS2 DEBUG] Planner control values:" << std::endl;
                    if (use_teleop_navigate_cmd_) {
                        std::cout << "  Input navigate_cmd: [" << navigate_cmd_from_teleop_[0] << ", " 
                                  << navigate_cmd_from_teleop_[1] << ", " << navigate_cmd_from_teleop_[2] << "]" << std::endl;
                        std::cout << "  Facing angle: " << planner_facing_angle_ << " rad (" 
                                  << (planner_facing_angle_ * 180.0 / M_PI) << " deg)" << std::endl;
                    }
                    std::cout << "  Base height command: " << base_height_command_ 
                              << " (clamped: " << std::clamp(base_height_command_, 0.1, 0.88) << ")" << std::endl;
                    std::cout << "  Final mode: " << final_mode << " (0=idle, 1=slow, 2=walk, 3=run, 4=squat, 6=kneel)" << std::endl;
                    std::cout << "  Final speed: " << final_speed << std::endl;
                    std::cout << "  Final height: " << final_height << std::endl;
                    std::cout << "  Movement direction: [" << final_movement[0] << ", " << final_movement[1] << ", " << final_movement[2] << "]" << std::endl;
                    std::cout << "  Facing direction: [" << final_facing_direction[0] << ", " << final_facing_direction[1] << ", " << final_facing_direction[2] << "]" << std::endl;
                }
            }

            // Update thread-safe buffer (single source of truth for planner thread)
            MovementState mode_state(final_mode, final_movement, final_facing_direction, final_speed, final_height);
            movement_state_buffer.SetData(mode_state);
        }
    }

    // Get ROS2 node for external use (e.g., spinning)
    std::shared_ptr<rclcpp::Node> get_node() const { return node_; }
    
    // Check if ROS2 node is healthy and receiving data
    bool is_receiving_control_goal_data() const { return received_control_goal_.load(); }
    bool is_ros2_ok() const { return rclcpp::ok() && node_ != nullptr; }
    
    // Reset data received flags (useful for debugging)
    void reset_data_flags() {
        received_control_goal_.store(false);
        last_control_goal_time_ns_.store(0);
    }

    // Get ROS timestamp in seconds (for state logging)
    // This is ROS2-specific and not part of the InputInterface base class
    double GetROSTimestamp() const {
        if (node_) {
            return node_->get_clock()->now().nanoseconds() / 1e9;
        }
        return 0.0;
    }

private:
    // ------------------------------------------------------------------
    // ROS 2 infrastructure
    // ------------------------------------------------------------------
    std::shared_ptr<rclcpp::Node> node_;  ///< Lightweight ROS 2 node for this handler.

    /// Subscription to `ControlPolicy/upper_body_pose` (ByteMultiArray, msgpack).
    rclcpp::Subscription<std_msgs::msg::ByteMultiArray>::SharedPtr control_goal_sub_;

    // ------------------------------------------------------------------
    // Thread-safe receiving buffer (written by callback, read by update())
    // ------------------------------------------------------------------
    ControlGoalMsg control_goal_buffer_;               ///< Latest deserialized message.
    std::mutex control_goal_mutex_;                     ///< Guards control_goal_buffer_.
    std::atomic<bool> received_control_goal_{false};   ///< True once at least one message arrived.
    std::atomic<int64_t> last_control_goal_time_ns_{0}; ///< Monotonic timestamp of last message (ns).
    static constexpr double CONTROL_GOAL_TIMEOUT = 1.0; ///< Seconds before a timeout reset.
    
    /// When true, use IK-processed transformation matrices for VR position;
    /// when false, use raw left_wrist / right_wrist matrices.
    bool use_ik_mode_ = false;

    // ------------------------------------------------------------------
    // Control-toggle state
    // ------------------------------------------------------------------
    bool control_is_active_ = false;       ///< Tracks the toggle state for toggle_policy_action.
    bool locomotion_mode_is_fast_ = false;  ///< false = SLOW_WALK (custom speed), true = WALK (default speed).
    
    // ------------------------------------------------------------------
    // Per-frame control flags (reset in update())
    // ------------------------------------------------------------------
    bool start_control_ = false;   ///< Start control this frame.
    bool stop_control_ = false;    ///< Stop control this frame.

    // ------------------------------------------------------------------
    // Teleop state (updated from control_goal_buffer_ in update())
    // ------------------------------------------------------------------
    std::array<double, 3> navigate_cmd_from_teleop_ = {0.0, 0.0, 0.0};  ///< [lin_x, lin_y, ang_z].
    bool use_teleop_navigate_cmd_ = false;   ///< True while navigate_cmd is valid.
    double base_height_command_ = 0.78;      ///< Thread-safe copy of base_height_command.
    
    /// Accumulated facing angle (radians), integrated from ang_vel_z each frame.
    double planner_facing_angle_ = 0.0;
    
    struct termios old_termios_;  ///< Saved terminal state for restoration on destruction.
    
    /**
     * @brief Deserialize a msgpack-encoded control-goal payload.
     * @param data  Raw bytes from the ByteMultiArray message.
     * @return Parsed ControlGoalMsg (valid == true on success).
     *
     * Expected keys: navigate_cmd, wrist_pose, left_wrist_after_ik,
     * right_wrist_after_ik, head_after_ik, left_wrist, right_wrist,
     * base_height_command, toggle_policy_action, locomotion_mode,
     * left_hand_joint, right_hand_joint, ros_timestamp.
     */
    ControlGoalMsg parse_msgpack_control_goal(const std::vector<uint8_t>& data) {
        ControlGoalMsg msg;
        msg.valid = false;
        
        try {
            // Use msgpack-c library for clean, efficient parsing
            msgpack::object_handle oh = msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size());
            msgpack::object deserialized = oh.get();
            
            // Convert to map
            if (deserialized.type != msgpack::type::MAP) {
                return msg;
            }
            
            std::map<std::string, msgpack::object> map_data;
            deserialized.convert(map_data);
            
            // Extract navigate_cmd (array of 3 doubles)
            if (map_data.count("navigate_cmd") && map_data["navigate_cmd"].type == msgpack::type::ARRAY) {
                auto nav_arr = map_data["navigate_cmd"].as<std::vector<double>>();
                if (nav_arr.size() >= 3) {
                    msg.navigate_cmd[0] = nav_arr[0];
                    msg.navigate_cmd[1] = nav_arr[1];
                    msg.navigate_cmd[2] = nav_arr[2];
                }
            }
            
            // Extract wrist_pose (array of 14 doubles)
            if (map_data.count("wrist_pose") && map_data["wrist_pose"].type == msgpack::type::ARRAY) {
                auto wrist_arr = map_data["wrist_pose"].as<std::vector<double>>();
                if (wrist_arr.size() >= 14) {
                    std::copy_n(wrist_arr.begin(), 14, msg.wrist_pose.begin());
                }
            }
            
            // Extract IK-processed matrices (16 doubles each)
            if (map_data.count("left_wrist_after_ik") && map_data["left_wrist_after_ik"].type == msgpack::type::ARRAY) {
                try {
                    auto nested_arr = map_data["left_wrist_after_ik"].as<std::vector<std::vector<double>>>();
                    if (nested_arr.size() == 4 && nested_arr[0].size() == 4) {
                        // Flatten 4x4 matrix to 1D array (row-major)
                        size_t idx = 0;
                        for (const auto& row : nested_arr) {
                            for (double val : row) {
                                msg.left_wrist_after_ik[idx++] = val;
                            }
                        }
                        msg.has_ik_data = true;
                    }
                } catch (const std::exception& e) {
                    if constexpr (DEBUG_LOGGING) {
                        std::cerr << "[ROS2 ERROR] Failed to parse left_wrist_after_ik: " << e.what() << std::endl;
                    }
                }
            }
            
            if (map_data.count("right_wrist_after_ik") && map_data["right_wrist_after_ik"].type == msgpack::type::ARRAY) {
                try {
                    auto nested_arr = map_data["right_wrist_after_ik"].as<std::vector<std::vector<double>>>();
                    if (nested_arr.size() == 4 && nested_arr[0].size() == 4) {
                        // Flatten 4x4 matrix to 1D array (row-major)
                        size_t idx = 0;
                        for (const auto& row : nested_arr) {
                            for (double val : row) {
                                msg.right_wrist_after_ik[idx++] = val;
                            }
                        }
                        msg.has_ik_data = true;
                    }
                } catch (const std::exception& e) {
                    if constexpr (DEBUG_LOGGING) {
                        std::cerr << "[ROS2 ERROR] Failed to parse right_wrist_after_ik: " << e.what() << std::endl;
                    }
                }
            }
            
            if (map_data.count("head_after_ik") && map_data["head_after_ik"].type == msgpack::type::ARRAY) {
                try {
                    auto nested_arr = map_data["head_after_ik"].as<std::vector<std::vector<double>>>();
                    if (nested_arr.size() == 4 && nested_arr[0].size() == 4) {
                        // Flatten 4x4 matrix to 1D array (row-major)
                        size_t idx = 0;
                        for (const auto& row : nested_arr) {
                            for (double val : row) {
                                msg.head_after_ik[idx++] = val;
                            }
                        }
                        msg.has_ik_data = true;
                    }
                } catch (const std::exception& e) {
                    if constexpr (DEBUG_LOGGING) {
                        std::cerr << "[ROS2 ERROR] Failed to parse head_after_ik: " << e.what() << std::endl;
                    }
                }
            }
            
            // Extract non-IK wrist matrices (16 doubles each)
            if (map_data.count("left_wrist") && map_data["left_wrist"].type == msgpack::type::ARRAY) {
                try {
                    auto nested_arr = map_data["left_wrist"].as<std::vector<std::vector<double>>>();
                    if (nested_arr.size() == 4 && nested_arr[0].size() == 4) {
                        // Flatten 4x4 matrix to 1D array (row-major)
                        size_t idx = 0;
                        for (const auto& row : nested_arr) {
                            for (double val : row) {
                                msg.left_wrist[idx++] = val;
                            }
                        }
                        msg.has_wrist_matrices = true;
                    }
                } catch (const std::exception& e) {
                    if constexpr (DEBUG_LOGGING) {
                        std::cerr << "[ROS2 ERROR] Failed to parse left_wrist: " << e.what() << std::endl;
                    }
                }
            }
            
            if (map_data.count("right_wrist") && map_data["right_wrist"].type == msgpack::type::ARRAY) {
                try {
                    auto nested_arr = map_data["right_wrist"].as<std::vector<std::vector<double>>>();
                    if (nested_arr.size() == 4 && nested_arr[0].size() == 4) {
                        // Flatten 4x4 matrix to 1D array (row-major)
                        size_t idx = 0;
                        for (const auto& row : nested_arr) {
                            for (double val : row) {
                                msg.right_wrist[idx++] = val;
                            }
                        }
                        msg.has_wrist_matrices = true;
                    }
                } catch (const std::exception& e) {
                    if constexpr (DEBUG_LOGGING) {
                        std::cerr << "[ROS2 ERROR] Failed to parse right_wrist: " << e.what() << std::endl;
                    }
                }
            }
            
            // Extract base_height_command (double)
            if (map_data.count("base_height_command")) {
                msg.base_height_command = map_data["base_height_command"].as<double>();
            }
            
            // Extract toggle_policy_action (bool)
            if (map_data.count("toggle_policy_action")) {
                msg.toggle_policy_action = map_data["toggle_policy_action"].as<bool>();
            }
            
            // Extract locomotion_mode (int: 0 = slow walk, 1 = fast walk)
            if (map_data.count("locomotion_mode")) {
                msg.locomotion_mode = map_data["locomotion_mode"].as<int>();
            }
            
            // Extract left_hand_joint (7 doubles - joint positions)
            if (map_data.count("left_hand_joint") && map_data["left_hand_joint"].type == msgpack::type::ARRAY) {
                auto left_hand_arr = map_data["left_hand_joint"].as<std::vector<double>>();
                if (left_hand_arr.size() >= 7) {
                    std::copy_n(left_hand_arr.begin(), 7, msg.left_hand_joint.begin());
                    msg.has_hand_joints = true;
                }
            }
            
            // Extract right_hand_joint (7 doubles - joint positions)
            if (map_data.count("right_hand_joint") && map_data["right_hand_joint"].type == msgpack::type::ARRAY) {
                auto right_hand_arr = map_data["right_hand_joint"].as<std::vector<double>>();
                if (right_hand_arr.size() >= 7) {
                    std::copy_n(right_hand_arr.begin(), 7, msg.right_hand_joint.begin());
                    msg.has_hand_joints = true;
                }
            }
            
            // Extract ros_timestamp (double - ROS time in seconds)
            if (map_data.count("ros_timestamp")) {
                msg.ros_timestamp = map_data["ros_timestamp"].as<double>();
            }
            
            msg.valid = true;
            
        } catch (const std::exception& e) {
            std::cerr << "[ROS2 ERROR] msgpack parsing failed: " << e.what() << std::endl;
            msg.valid = false;
        }
        
        return msg;
    }
    
    /**
     * @brief Extract the 3×3 rotation matrix from a flattened row-major 4×4 transform.
     * @return 3×3 rotation matrix compatible with math_utils::rotation_matrix_to_quat_d().
     */
    std::array<std::array<double, 3>, 3> extract_rotation_from_transform(const std::array<double, 16>& transform) {
        // 4x4 matrix in row-major: [R00,R01,R02,tx, R10,R11,R12,ty, R20,R21,R22,tz, 0,0,0,1]
        return {{
            {transform[0], transform[1], transform[2]},    // Row 0
            {transform[4], transform[5], transform[6]},    // Row 1
            {transform[8], transform[9], transform[10]}    // Row 2
        }};
    }
    
    /// Extract the translation vector [x, y, z] from a flattened row-major 4×4 transform.
    std::array<double, 3> extract_position_from_transform(const std::array<double, 16>& transform) {
        // 4x4 matrix in row-major: position is at [3, 7, 11]
        return {transform[3], transform[7], transform[11]};
    }
    
    /// Multiply a 3×3 rotation matrix by a 3D offset vector (R × v).
    std::array<double, 3> apply_rotation_to_offset(
        const std::array<std::array<double, 3>, 3>& rot_mat, 
        const std::array<double, 3>& offset) {
        return {
            rot_mat[0][0] * offset[0] + rot_mat[0][1] * offset[1] + rot_mat[0][2] * offset[2],  // Row 0 * offset
            rot_mat[1][0] * offset[0] + rot_mat[1][1] * offset[1] + rot_mat[1][2] * offset[2],  // Row 1 * offset
            rot_mat[2][0] * offset[0] + rot_mat[2][1] * offset[1] + rot_mat[2][2] * offset[2]   // Row 2 * offset
        };
    }
    
    /**
     * @brief Quantise an angle to the nearest 45° bin and return a direction-dependent speed.
     *
     * The 8 bins: 0° (forward), ±45° (forward-diagonal), ±90° (lateral),
     * ±135° (backward-diagonal), 180° (backward).
     *
     * @param angle  Input angle in radians (will be normalised to [−π, π]).
     * @return {binned_angle, slow_walk_speed} – angle snapped to nearest bin,
     *         and the corresponding speed for SLOW_WALK mode (faster forward,
     *         slower backward).
     */
    std::pair<double, double> bin_angle_to_8_directions(double angle) {
        constexpr double BIN_SIZE = M_PI / 4.0;  // 45 degrees in radians
        constexpr int NUM_BINS = 8;
        
        // Normalize angle to [-π, π]
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        
        // Find nearest bin
        int bin_index = static_cast<int>(std::round(angle / BIN_SIZE));
        
        // Handle wrap-around (bin_index can be -4 to 4)
        if (bin_index > 4) bin_index -= NUM_BINS;
        if (bin_index < -4) bin_index += NUM_BINS;
        
        // Convert back to angle
        double binned_angle = bin_index * BIN_SIZE;
        
        // Determine speed based on direction bin for slow walk mode
        // Faster forward/lateral, slower backward
        double slow_walk_speed;
        switch (bin_index) {
            case 0:   // Forward (0°)
            case 1:   // Forward-right (45°)
            case -1:  // Forward-left (-45°)
                slow_walk_speed = 0.3f;
                break;
            case 2:   // Right (90°)
            case -2:  // Left (-90°)
                slow_walk_speed = 0.35f;
                break;
            case 3:   // Back-right (135°)
            case -3:  // Back-left (-135°)
                slow_walk_speed = 0.25f;
                break;
            case 4:   // Backward (180°)
            case -4:  // Backward (-180°)
                slow_walk_speed = 0.2f;
                break;
            default:
                slow_walk_speed = 0.2f;  // Fallback
                break;
        }
        
        return {binned_angle, slow_walk_speed};
    }

    /**
     * @brief ROS 2 subscriber callback (runs on the executor thread).
     *
     * Deserialises the ByteMultiArray payload via msgpack, stores the result
     * in `control_goal_buffer_` under `control_goal_mutex_`, and sets the
     * `received_control_goal_` flag.  Edge-triggered commands (toggle_policy_action)
     * are accumulated with OR logic to prevent lost pulses.
     */
    void control_goal_callback(std::shared_ptr<const std_msgs::msg::ByteMultiArray> msg) {
        try {
            // Convert ByteMultiArray to vector<uint8_t>
            // msg->data is already std::vector<uint8_t>, so just copy it
            std::vector<uint8_t> data(msg->data.begin(), msg->data.end());
            
            // Parse msgpack data
            ControlGoalMsg goal_msg = parse_msgpack_control_goal(data);
            
            if (goal_msg.valid) {
                // Store in receiving buffer (thread-safe)
                {
                    std::lock_guard<std::mutex> lock(control_goal_mutex_);
                    // Edge-triggered commands: accumulate with OR (like mode_control)
                    // This prevents losing a toggle if it arrives between update() cycles
                    bool prev_toggle_policy = control_goal_buffer_.toggle_policy_action;
                    control_goal_buffer_ = goal_msg;
                    control_goal_buffer_.toggle_policy_action = prev_toggle_policy || goal_msg.toggle_policy_action;
                    // locomotion_mode is a direct state value (0 or 1), not accumulated
                }
                received_control_goal_.store(true);
                // Update timestamp for timeout tracking (using steady_clock for monotonic timing)
                last_control_goal_time_ns_.store(std::chrono::steady_clock::now().time_since_epoch().count());
                
                if constexpr (DEBUG_LOGGING) {
                    static int goal_counter = 0;
                    goal_counter++;
                    if (goal_counter % 50 == 0) {  // Log every 50 messages to avoid spam
                        std::cout << "[ROS2 DEBUG] Control goal message received:" << std::endl;
                        std::cout << "  navigate_cmd: [" << goal_msg.navigate_cmd[0] << ", " 
                                  << goal_msg.navigate_cmd[1] << ", " << goal_msg.navigate_cmd[2] << "]" << std::endl;
                        std::cout << "  base_height: " << goal_msg.base_height_command << std::endl;
                    }
                }
            } else {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 DEBUG] Invalid control goal message received" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 ERROR] Failed to process control goal message: " << e.what() << std::endl;
            }
        }
    }

    // Helper method to initialize ROS2 subscriber
    void setup_subscribers() {
        // Create subscriber for control goal topic (teleop commands from Python)
        control_goal_sub_ = node_->create_subscription<std_msgs::msg::ByteMultiArray>(
            "ControlPolicy/upper_body_pose",
            1,  // QoS depth = 1 for some buffering
            [this](std::shared_ptr<const std_msgs::msg::ByteMultiArray> msg) {
                this->control_goal_callback(msg);
            }
        );
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ROS2 DEBUG] Control goal subscriber created for topic: ControlPolicy/upper_body_pose" << std::endl;
        }
    }

    /*
     * ROS2 Topic Structure:
     * 
     * ControlPolicy/upper_body_pose (std_msgs/ByteMultiArray) - msgpack-serialized ControlGoalMsg:
     *    - navigate_cmd: double[3] (navigation velocities [lin_vel_x, lin_vel_y, ang_vel_z])
     *    - wrist_pose: double[14] (left + right wrist poses: [x,y,z,qw,qx,qy,qz] * 2)
     *    - left_wrist_after_ik: double[16] (4x4 transformation matrix, flattened row-major)
     *    - right_wrist_after_ik: double[16] (4x4 transformation matrix, flattened row-major)
     *    - head_after_ik: double[16] (4x4 transformation matrix, flattened row-major)
     *    - left_wrist: double[16] (4x4 transformation matrix, flattened row-major, non-IK)
     *    - right_wrist: double[16] (4x4 transformation matrix, flattened row-major, non-IK)
     *    - left_hand_joint: double[7] (7 DOF joint positions for left hand)
     *    - right_hand_joint: double[7] (7 DOF joint positions for right hand)
     *    - base_height_command: double (desired base height)
     *    - toggle_policy_action: bool (toggle between start/stop control)
     *    - locomotion_mode: int (0 = slow walk with custom speed, 1 = fast walk with default speed)
     *    - ros_timestamp: double (ROS time in seconds for synchronization)
     *    - valid: bool (message validity flag)
     */
};

#endif // HAS_ROS2

#endif // ROS2_INPUT_HANDLER_HPP
