/**
 * @file ros2_output_handler.hpp
 * @brief ROS 2 output handler for publishing robot state and configuration.
 *
 * ROS2OutputHandler creates a lightweight ROS 2 node and publishes two topics:
 *
 *   Topic                        | QoS              | Description
 *   -----------------------------|------------------|----------------------------
 *   G1Env/env_state_act          | depth=10         | Per-tick robot state + action data (msgpack).
 *   WBCPolicy/robot_config       | transient_local  | One-shot robot configuration (msgpack).
 *
 * ## State Topic (`G1Env/env_state_act`)
 *
 * Published every control-loop tick.  The msgpack map contains:
 *
 *   Key                  | Type         | Description
 *   ---------------------|--------------|------------
 *   control_loop_type    | string       | Always "cpp".
 *   index                | int          | Monotonic state-logger entry index.
 *   ros_timestamp        | double       | ROS 2 time in seconds.
 *   base_quat            | double[4]    | IMU base quaternion (w,x,y,z).
 *   base_ang_vel         | double[3]    | Base angular velocity.
 *   body_torso_quat      | double[4]    | Torso IMU quaternion.
 *   body_torso_ang_vel   | double[3]    | Torso angular velocity.
 *   body_q               | double[29]   | Joint positions (MuJoCo order + default offsets).
 *   body_dq              | double[29]   | Joint velocities (MuJoCo order).
 *   last_action          | double[29]   | Last policy action (MuJoCo order, scaled + offset).
 *   left_hand_q          | double[7]    | Left-hand joint positions.
 *   left_hand_dq         | double[7]    | Left-hand joint velocities.
 *   right_hand_q         | double[7]    | Right-hand joint positions.
 *   right_hand_dq        | double[7]    | Right-hand joint velocities.
 *   last_left_hand_action| double[7]    | Last left-hand action.
 *   last_right_hand_action| double[7]   | Last right-hand action.
 *   token_state          | double[N]    | Encoder token state (empty if not available).
 *   init_base_quat       | double[4]    | Initial base quaternion (if heading state available).
 *   delta_heading        | double       | Delta heading (if heading state available).
 *
 * ## Config Topic (`WBCPolicy/robot_config`)
 *
 * Published once at startup via `publish_config()`.  Uses `transient_local`
 * QoS so that late-joining subscribers still receive it.  Contains the
 * StateLogger configuration map (policy parameters, joint mappings, etc.).
 *
 * ## Thread Safety
 *
 * `publish()` is called from the control thread.  The ROS 2 node does not
 * run its own executor – `spin_some()` is not called here (unlike the input
 * handler).
 */

#ifndef ROS2_OUTPUT_HANDLER_HPP
#define ROS2_OUTPUT_HANDLER_HPP

#if HAS_ROS2
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/exceptions/exceptions.hpp>
#include <std_msgs/msg/byte_multi_array.hpp>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <map>
#include <vector>
#include <variant>
#include <stdexcept>
#include <msgpack.hpp>

#include "output_interface.hpp"
#include "../policy_parameters.hpp"  // For isaaclab_to_mujoco, default_angles, g1_action_scale
#include "../robot_parameters.hpp"  // For HeadingState
#include "../utils.hpp"  // For DataBuffer

/**
 * @class ROS2OutputHandler
 * @brief OutputInterface that publishes robot state and configuration over ROS 2 topics.
 *
 * Owns a ROS 2 node with two publishers (state + config).  The config topic
 * uses `transient_local` QoS so late joiners receive it automatically.
 */
class ROS2OutputHandler : public OutputInterface {
public:
    /// Compile-time toggle for debug log output.
    static constexpr bool DEBUG_LOGGING = true;

    /**
     * @brief Construct the handler: create a ROS 2 node and advertise publishers.
     * @param logger     Reference to the shared StateLogger.
     * @param node_name  ROS 2 node name (default: "g1_output_handler").
     * @throws std::exception if ROS 2 initialisation or node creation fails.
     */
    explicit ROS2OutputHandler(StateLogger& logger, const std::string& node_name = "g1_output_handler") 
        : OutputInterface(logger) {
        // Initialize ROS2 if not already initialized
        if (!rclcpp::ok()) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 Output DEBUG] Initializing ROS2" << std::endl;
            }
            rclcpp::init(0, nullptr);
        }
        
        try {
            // Initialize ROS2 node
            node_ = rclcpp::Node::make_shared(node_name);
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 Output DEBUG] ROS2OutputHandler node '" << node_name << "' initialized" << std::endl;
            }
            
            // Setup publishers
            setup_publishers();
            
        } catch (const std::exception& e) {
            std::cerr << "[ROS2 Output ERROR] Failed to initialize ROS2OutputHandler: " << e.what() << std::endl;
            throw;
        }
        
        type_ = OutputType::ROS2;
    }

    /// Destructor – tears down publishers before the node to avoid DDS assertion errors.
    ~ROS2OutputHandler() {
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ROS2 Output DEBUG] ROS2OutputHandler destructor called" << std::endl;
        }
        
        // CRITICAL: Proper shutdown order to avoid EntityDelegate assertion
        try {
            // Step 1: Reset publishers BEFORE resetting node
            if (state_logger_pub_) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 Output DEBUG] Resetting state logger publisher" << std::endl;
                }
                state_logger_pub_.reset();
            }
            
            if (robot_config_pub_) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 Output DEBUG] Resetting robot config publisher" << std::endl;
                }
                robot_config_pub_.reset();
            }
            
            // Step 2: Allow DDS cleanup time
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Step 3: Reset node AFTER publishers are cleaned up
            if (node_) {
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ROS2 Output DEBUG] Resetting ROS2 node" << std::endl;
                }
                node_.reset();
            }
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 Output DEBUG] ROS2OutputHandler cleanup complete" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ROS2 Output ERROR] Exception during ROS2OutputHandler cleanup: " << e.what() << std::endl;
        }
    }

    /// Publish the latest robot state to `G1Env/env_state_act` (called each control tick).
    void publish(
        const std::array<double, 9>& vr_3point_position,
        const std::array<double, 12>& vr_3point_orientation,
        const std::array<double, 3>& vr_3point_compliance,
        const std::array<double, 7>& left_hand_joint,
        const std::array<double, 7>& right_hand_joint,
        const std::array<double, 4>& init_ref_data_root_rot_array,
        DataBuffer<HeadingState>& heading_state_buffer,
        std::shared_ptr<const MotionSequence> current_motion,
        int current_frame
    ) override {
        if (!node_ || !rclcpp::ok()) {
            return;
        }
        
        // Config should already be published via publish_config() during initialization
        // If it wasn't, that's an error condition (should have failed at init)
        if (!robot_config_published_) {
            std::cerr << "[ROS2 Output ERROR] Robot config was not published during initialization!" << std::endl;
            // Don't try to publish here - fail fast was supposed to happen at init
            return;
        }
        
        // Publish state logger data (includes token_state from LogPostState)
        if (state_logger_pub_) {
            try {
                publish_state_logger_state(heading_state_buffer);
            } catch (const std::exception& e) {
                if constexpr (DEBUG_LOGGING) {
                    std::cerr << "[ROS2 Output ERROR] Failed to publish state: " << e.what() << std::endl;
                }
            }
        }
    }

    /// @return The ROS 2 node (e.g. for external spinning).
    std::shared_ptr<rclcpp::Node> get_node() const { return node_; }
    
    /// @return True if the ROS 2 context and node are both healthy.
    bool is_ros2_ok() const { return rclcpp::ok() && node_ != nullptr; }
    
    /// @return Current ROS 2 clock time in seconds (for state-logger timestamps).
    double GetROSTimestamp() const {
        if (node_) {
            return node_->get_clock()->now().nanoseconds() / 1e9;
        }
        return 0.0;
    }
    
    /**
     * @brief Publish the robot configuration to `WBCPolicy/robot_config`.
     *
     * Typically called once after initialisation (before the control loop starts).
     * Uses `transient_local` QoS so that late-joining subscribers still receive it.
     *
     * @throws std::runtime_error if the node is not ready or the config is empty.
     *
     * @note All config fields are set before StateLogger construction and are
     *       immutable, so they're guaranteed to be available when this is called.
     */
    void publish_config() {
        if (!node_ || !rclcpp::ok()) {
            throw std::runtime_error("[ROS2 Output ERROR] Cannot publish config: ROS2 node not ready");
        }
        if (robot_config_published_) {
            return;  // Already published
        }
        
        // Give ROS2 a moment to advertise the publisher (needed for DDS discovery)
        // This ensures subscribers can discover the publisher before we publish
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto config_opt = state_logger_.GetConfig();
        if (!config_opt.has_value() || config_opt->empty()) {
            throw std::runtime_error("[ROS2 Output ERROR] Cannot publish config: StateLogger config is empty or unavailable");
        }
        
        PublishRobotConfig(*config_opt);
        robot_config_published_ = true;
        std::cout << "[ROS2 Output] Published robot config to WBCPolicy/robot_config (immediate)" << std::endl;
    }

private:
    // ------------------------------------------------------------------
    // ROS 2 infrastructure
    // ------------------------------------------------------------------
    std::shared_ptr<rclcpp::Node> node_;  ///< Lightweight ROS 2 node for publishing.

    /// Publisher for per-tick robot state (G1Env/env_state_act).
    rclcpp::Publisher<std_msgs::msg::ByteMultiArray>::SharedPtr state_logger_pub_;
    /// Publisher for one-shot robot config (WBCPolicy/robot_config, transient_local).
    rclcpp::Publisher<std_msgs::msg::ByteMultiArray>::SharedPtr robot_config_pub_;
    
    bool robot_config_published_ = false;  ///< True once publish_config() has succeeded.

    /// Create the two ROS 2 publishers with appropriate QoS settings.
    void setup_publishers() {
        // Create publisher for state logger data (robot state + actions)
        state_logger_pub_ = node_->create_publisher<std_msgs::msg::ByteMultiArray>(
            "G1Env/env_state_act",
            10  // QoS depth = 10 for buffering
        );
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ROS2 Output DEBUG] Created publisher for G1Env/env_state_act" << std::endl;
        }

        // Create publisher for robot configuration (with transient_local QoS for late joiners)
        // This ensures that subscribers that connect after the message is published will still receive it
        rclcpp::QoS config_qos(1);  // depth = 1 since we only publish once
        config_qos.transient_local();  // Keep last message for late joiners
        config_qos.reliable();  // Ensure delivery
        
        robot_config_pub_ = node_->create_publisher<std_msgs::msg::ByteMultiArray>(
            "WBCPolicy/robot_config",
            config_qos
        );
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ROS2 Output DEBUG] Created publisher for WBCPolicy/robot_config (transient_local)" << std::endl;
        }
    }

    /// Serialise the robot configuration map to msgpack and publish on robot_config_pub_.
    void PublishRobotConfig(const std::map<std::string, std::variant<std::string, int, double, bool>>& config) {
        if (!robot_config_pub_) {
            if constexpr (DEBUG_LOGGING) {
                std::cerr << "[ROS2 Output ERROR] Robot config publisher not initialized" << std::endl;
            }
            return;
        }
        
        try {
            // Pack config into msgpack using sbuffer
            msgpack::sbuffer sbuf;
            msgpack::packer<msgpack::sbuffer> pk(&sbuf);
            
            // Pack as map
            pk.pack_map(config.size());
            for (const auto& [key, value] : config) {
                pk.pack(key);
                // Pack value based on type
                std::visit([&pk](auto&& arg) {
                    pk.pack(arg);
                }, value);
            }
            
            // Convert sbuffer to vector for ROS2 message
            std_msgs::msg::ByteMultiArray msg;
            msg.data.assign(sbuf.data(), sbuf.data() + sbuf.size());
            
            robot_config_pub_->publish(msg);
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ROS2 Output DEBUG] Published robot config with " << config.size() << " fields" << std::endl;
            }
        } catch (const std::exception& e) {
            if constexpr (DEBUG_LOGGING) {
                std::cerr << "[ROS2 Output ERROR] Failed to publish robot config: " << e.what() << std::endl;
            }
        }
    }

    /// Publish the latest state-logger entry to `G1Env/env_state_act`.
    void publish_state_logger_state(const DataBuffer<HeadingState>& heading_state_buffer) {
        std::vector<uint8_t> msgpack_data = parse_state_logger_state_to_msgpack(heading_state_buffer);
        
        // Skip publishing if no data available
        if (msgpack_data.empty()) {
            return;
        }
        
        // Convert to ROS2 ByteMultiArray message
        std_msgs::msg::ByteMultiArray msg;
        msg.data = msgpack_data;
        
        state_logger_pub_->publish(msg);
    }

    /**
     * @brief Serialise the latest state-logger entry to a msgpack byte vector.
     *
     * Applies IsaacLab → MuJoCo joint reordering, adds default-angle offsets
     * for positions, and scales + offsets actions.  Optionally appends heading
     * state (init_base_quat, delta_heading) if available.
     *
     * @return Serialised msgpack bytes, or empty vector if no data is available.
     */
    std::vector<uint8_t> parse_state_logger_state_to_msgpack(const DataBuffer<HeadingState>& heading_state_buffer) {
        // Check if logger has actual data (not just zero-padded entries)
        if (state_logger_.size() == 0) {
            return std::vector<uint8_t>();  // Return empty if no actual data
        }
        
        // Get the latest state from logger
        std::vector<StateLogger::Entry> entries = state_logger_.GetLatest(1);
        const StateLogger::Entry& state = entries[0];
        
        // Pack all fields into msgpack map using sbuffer
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);
        
        // Get heading state if available
        HeadingState heading_state;
        bool has_heading_state = false;
        auto heading_state_data = heading_state_buffer.GetDataWithTime().data;
        if (heading_state_data) {
            heading_state = *heading_state_data;
            has_heading_state = true;
        }
        
        // Create a map with all entry fields (17 base fields + 2 optional heading fields)
        int num_fields = 17;
        if (has_heading_state) {
            num_fields += 2;  // Add init_base_quat and delta_heading
        }
        pk.pack_map(num_fields);
        
        // Pack control loop type identifier
        pk.pack("control_loop_type");
        pk.pack("cpp");
        
        // Pack index
        pk.pack("index");
        pk.pack(state.index);
        
        // Pack ros_timestamp (as seconds, using ROS2 time stored in entry to align with Python)
        pk.pack("ros_timestamp");
        pk.pack(state.ros_timestamp);
        
        // Pack base_quat (4 doubles: qw, qx, qy, qz)
        pk.pack("base_quat");
        pk.pack_array(4);
        for (const auto& val : state.base_quat) {
            pk.pack(val);
        }
        
        // Pack base_ang_vel (3 doubles: wx, wy, wz)
        pk.pack("base_ang_vel");
        pk.pack_array(3);
        for (const auto& val : state.base_ang_vel) {
            pk.pack(val);
        }
        
        // Pack body_torso_quat (4 doubles: qw, qx, qy, qz)
        pk.pack("body_torso_quat");
        pk.pack_array(4);
        for (const auto& val : state.body_torso_quat) {
            pk.pack(val);
        }
        
        // Pack body_torso_ang_vel (3 doubles: wx, wy, wz)
        pk.pack("body_torso_ang_vel");
        pk.pack_array(3);
        for (const auto& val : state.body_torso_ang_vel) {
            pk.pack(val);
        }
        
        // Pack body_q (dynamic size) - recover original values (add offset, convert to MuJoCo order)
        pk.pack("body_q");
        pk.pack_array(state.body_q.size());
        if (state.body_q.size() == 29) {
            // Create temporary array for MuJoCo-ordered values
            std::array<double, 29> body_q_mujoco;
            for (size_t i = 0; i < 29; ++i) {
                // Add back the offset (body_q is in IsaacLab order, default_angles is in MuJoCo order)
                body_q_mujoco[i] = state.body_q[isaaclab_to_mujoco[i]] + default_angles[i];
            }
            // Pack in MuJoCo order
            for (const auto& val : body_q_mujoco) {
                pk.pack(val);
            }
        } else {
            // Fallback: pack as-is if size doesn't match expected
            for (const auto& val : state.body_q) {
                pk.pack(val);
            }
        }
        
        // Pack body_dq (dynamic size) - convert to MuJoCo order (no offset for velocities)
        pk.pack("body_dq");
        pk.pack_array(state.body_dq.size());
        if (state.body_dq.size() == 29) {
            // Create temporary array for MuJoCo-ordered values
            std::array<double, 29> body_dq_mujoco;
            for (size_t i = 0; i < 29; ++i) {
                // Convert to MuJoCo order (velocities don't have offset)
                body_dq_mujoco[i] = state.body_dq[isaaclab_to_mujoco[i]];
            }
            // Pack in MuJoCo order
            for (const auto& val : body_dq_mujoco) {
                pk.pack(val);
            }
        } else {
            // Fallback: pack as-is if size doesn't match expected
            for (const auto& val : state.body_dq) {
                pk.pack(val);
            }
        }
        
        // Pack last_action (dynamic size) - convert to MuJoCo order
        pk.pack("last_action");
        pk.pack_array(state.last_action.size());
        if (state.last_action.size() == 29) {
            // Create temporary array for MuJoCo-ordered values
            std::array<double, 29> last_action_mujoco;
            for (size_t i = 0; i < 29; ++i) {
                // Convert to MuJoCo order (actions are in IsaacLab order in logger)
                last_action_mujoco[i] = state.last_action[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i];
            }
            // Pack in MuJoCo order
            for (const auto& val : last_action_mujoco) {
                pk.pack(val);
            }
        } else {
            // Fallback: pack as-is if size doesn't match expected
            for (const auto& val : state.last_action) {
                pk.pack(val);
            }
        }
        
        // Pack left_hand_q (7 doubles)
        pk.pack("left_hand_q");
        pk.pack_array(state.left_hand_q.size());
        for (const auto& val : state.left_hand_q) {
            pk.pack(val);
        }
        
        // Pack left_hand_dq (7 doubles)
        pk.pack("left_hand_dq");
        pk.pack_array(state.left_hand_dq.size());
        for (const auto& val : state.left_hand_dq) {
            pk.pack(val);
        }
        
        // Pack right_hand_q (7 doubles)
        pk.pack("right_hand_q");
        pk.pack_array(state.right_hand_q.size());
        for (const auto& val : state.right_hand_q) {
            pk.pack(val);
        }
        
        // Pack right_hand_dq (7 doubles)
        pk.pack("right_hand_dq");
        pk.pack_array(state.right_hand_dq.size());
        for (const auto& val : state.right_hand_dq) {
            pk.pack(val);
        }
        
        // Pack last_left_hand_action (7 doubles)
        pk.pack("last_left_hand_action");
        pk.pack_array(state.last_left_hand_action.size());
        for (const auto& val : state.last_left_hand_action) {
            pk.pack(val);
        }
        
        // Pack last_right_hand_action (7 doubles)
        pk.pack("last_right_hand_action");
        pk.pack_array(state.last_right_hand_action.size());
        for (const auto& val : state.last_right_hand_action) {
            pk.pack(val);
        }
        
        // Pack token_state (dynamic size, post-state data from encoder)
        pk.pack("token_state");
        if (state.has_post_state_data && !state.token_state.empty()) {
            pk.pack_array(state.token_state.size());
            for (const auto& val : state.token_state) {
                pk.pack(val);
            }
        } else {
            // Pack empty array if no token state available
            pk.pack_array(0);
        }
        
        // Pack init_base_quat and delta_heading from heading state (if available)
        if (has_heading_state) {
            // Pack init_base_quat (4 doubles: qw, qx, qy, qz)
            pk.pack("init_base_quat");
            pk.pack_array(4);
            for (const auto& val : heading_state.init_base_quat) {
                pk.pack(val);
            }
            
            // Pack delta_heading (single double)
            pk.pack("delta_heading");
            pk.pack(heading_state.delta_heading);
        }
        
        // Convert sbuffer to vector
        std::vector<uint8_t> buffer(sbuf.data(), sbuf.data() + sbuf.size());
        return buffer;
    }
};

#endif // HAS_ROS2

#endif // ROS2_OUTPUT_HANDLER_HPP

