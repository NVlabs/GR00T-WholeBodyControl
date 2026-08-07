/**
 * @file zmq_output_handler.hpp
 * @brief ZMQ PUB output handler for publishing robot state and configuration.
 *
 * Publishes two topic-prefixed streams over a single ZMQ PUB socket:
 *
 *   Topic Prefix    | Frequency       | Description
 *   ----------------|-----------------|-------------------------------------------
 *   {user_topic}    | Every tick      | Combined state + visualisation (msgpack).
 *   robot_config    | Every ~2 s      | Robot configuration (msgpack, always re-published).
 *
 * Wire format (single-part ZMQ message):
 *
 *   [topic_prefix][msgpack payload]
 *
 * The topic prefix is a plain string (e.g. "g1_debug") prepended to the
 * message so that subscribers using `zmq::sockopt::subscribe` can filter by
 * topic.
 *
 * ---------------------------------------------------------------------------
 * ## `{user_topic}` (e.g. `g1_debug`) — published every tick
 * ---------------------------------------------------------------------------
 *
 * A single msgpack map with up to 30 keys (28 always-present + 2 conditional).
 * All joints are in **MuJoCo order** (remapped from IsaacLab via
 * `isaaclab_to_mujoco`).
 *
 *   #  | Key                    | Type         | Description
 *   ---|------------------------|--------------|---------------------------------------------
 *      | **Metadata**           |              |
 *   1  | control_loop_type      | string       | Always "cpp".
 *   2  | index                  | int          | Monotonic state-logger entry index.
 *   3  | ros_timestamp          | double       | ROS 2 wall-clock (s); 0.0 if no ROS 2.
 *      |                        |              |
 *      | **Base IMU**           |              |
 *   4  | base_quat              | double[4]    | Base IMU quaternion (w,x,y,z).
 *   5  | base_ang_vel           | double[3]    | Base angular velocity.
 *   6  | body_torso_quat        | double[4]    | Torso IMU quaternion.
 *   7  | body_torso_ang_vel     | double[3]    | Torso angular velocity.
 *      |                        |              |
 *      | **Body joints**        |              |
 *   8  | body_q                 | double[29]   | Joint positions (+ default offsets).
 *   9  | body_dq                | double[29]   | Joint velocities.
 *      |                        |              |
 *      | **Hand joints**        |              |
 *  10  | left_hand_q            | double[7]    | Left-hand joint positions (from state logger).
 *  11  | left_hand_dq           | double[7]    | Left-hand joint velocities.
 *  12  | right_hand_q           | double[7]    | Right-hand joint positions (from state logger).
 *  13  | right_hand_dq          | double[7]    | Right-hand joint velocities.
 *      |                        |              |
 *      | **Policy actions**     |              |
 *  14  | last_action            | double[29]   | Last body action (scaled + default offsets).
 *  15  | last_left_hand_action  | double[7]    | Last left-hand action.
 *  16  | last_right_hand_action | double[7]    | Last right-hand action.
 *      |                        |              |
 *      | **Encoder**            |              |
 *  17  | token_state            | double[N]    | Encoder token state (empty array if N/A).
 *      |                        |              |
 *      | **Heading** *(conditional — only when heading state is available)* |
 *  18  | init_base_quat         | double[4]    | Initial base quaternion at heading init.
 *  19  | delta_heading          | double       | Accumulated heading delta (rad).
 *      |                        |              |
 *      | **Viz: targets** *(from current motion frame + heading correction)* |
 *  20  | base_trans_target      | double[3]    | Target base translation.
 *  21  | base_quat_target       | double[4]    | Target base quaternion.
 *  22  | body_q_target          | double[29]   | Target joint positions.
 *      |                        |              |
 *      | **Viz: measured**      |              |
 *  23  | base_trans_measured    | double[3]    | Measured base translation (fixed default).
 *  24  | base_quat_measured     | double[4]    | Measured base quaternion (= base_quat).
 *  25  | body_q_measured        | double[29]   | Measured joint positions (= body_q).
 *  26  | left_hand_q_measured   | double[7]    | Measured left-hand Dex3 positions.
 *  27  | right_hand_q_measured  | double[7]    | Measured right-hand Dex3 positions.
 *      |                        |              |
 *      | **Viz: VR 3-point**    |              |
 *  28  | vr_3point_position     | double[9]    | VR positions (3×xyz, target body frame).
 *  29  | vr_3point_orientation  | double[12]   | VR orientations (3×quat wxyz).
 *  30  | vr_3point_compliance   | double[3]    | VR compliance (left arm, right arm, head).
 *
 * ---------------------------------------------------------------------------
 * ## `robot_config` — re-published every ~2 s
 * ---------------------------------------------------------------------------
 *
 * StateLogger configuration map (policy parameters, joint mappings, etc.)
 * as a msgpack map of string → string | int | double | bool.
 * Re-published on every `publish()` tick (throttled to ~2 s intervals) so
 * that late-joining subscribers always receive it (ZMQ PUB has no persistence).
 *
 * ---------------------------------------------------------------------------
 * ## Socket Options
 * ---------------------------------------------------------------------------
 *
 * - Send HWM = 10 (old messages dropped rather than queued).
 * - Send buffer = 32 KB.
 * - Linger = 0 (immediate close, no pending-send wait).
 * - Non-blocking send (dontwait) to avoid stalling the control loop.
 */

#ifndef ZMQ_OUTPUT_HANDLER_HPP
#define ZMQ_OUTPUT_HANDLER_HPP

#include <memory>
#include <iostream>
#include <chrono>
#include <cstring>
#include <map>
#include <vector>
#include <variant>
#include <stdexcept>
#include <zmq.hpp>
#include <msgpack.hpp>

#include "output_interface.hpp"
#include "../policy_parameters.hpp"  // For isaaclab_to_mujoco, default_angles, g1_action_scale
#include "../robot_parameters.hpp"   // For HeadingState
#include "../utils.hpp"              // For DataBuffer

/**
 * @class ZMQOutputHandler
 * @brief OutputInterface that publishes state data over a ZMQ PUB socket.
 */
class ZMQOutputHandler : public OutputInterface {
public:
    static constexpr bool DEBUG_LOGGING = true;

    /**
     * @brief Construct the handler: create a ZMQ PUB socket and bind to the given port.
     * @param logger  Reference to the shared StateLogger.
     * @param port    TCP port to bind the PUB socket to (e.g. 5557).
     * @param topic   Topic prefix prepended to each published message.
     */
    explicit ZMQOutputHandler(StateLogger& logger, int port, const std::string& topic) 
        : OutputInterface(logger), realtime_debug_context_(1), topic_(topic),
          robot_config_topic_("robot_config") {

        std::cout << "Initializing realtime debug socket" << std::endl;
        std::cout << "Binding to port: " << port << " and topic: " << topic_ << std::endl;
        realtime_debug_socket_ = std::make_unique<zmq::socket_t>(realtime_debug_context_, ZMQ_PUB);

        realtime_debug_socket_->set(zmq::sockopt::sndhwm, 10);     // Drop old messages quickly
        realtime_debug_socket_->set(zmq::sockopt::sndbuf, 32768);   // 32 KB send buffer
        realtime_debug_socket_->set(zmq::sockopt::linger, 0);       // No lingering on close
        realtime_debug_socket_->bind("tcp://*:" + std::to_string(port));

        std::cout << "[INFO] Realtime debug socket bound to port: " << port << std::endl;

        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ZMQ Output DEBUG] ZMQOutputHandler initialized with topics: "
                      << "'" << topic_ << "' (combined state+viz), "
                      << "'" << robot_config_topic_ << "' (config)" << std::endl;
        }
        
        type_ = OutputType::ZMQ;
    }

    /**
     * @brief Send combined visualisation + state-logger data in a single ZMQ message (non-blocking, called each tick).
     *
     * Calls `create_output_data_map()` to compute visualisation targets, then
     * `pack_combined_state()` to merge state-logger + viz fields into one msgpack
     * buffer, and sends it on the user topic with `zmq::send_flags::dontwait`.
     */
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
    ) override
    {
        // 1. Compute visualisation data (populates output_data_map_)
        create_output_data_map(
            vr_3point_position,
            vr_3point_orientation,
            vr_3point_compliance,
            left_hand_joint,
            right_hand_joint,
            init_ref_data_root_rot_array,
            heading_state_buffer,
            current_motion,
            current_frame
        );

        // 2. Build a single combined message with state-logger + visualisation fields
        pack_combined_state(heading_state_buffer);

        // 3. Send once on user topic (e.g. "g1_debug")
        if (state_data_sbuf_.size() > 0) {
            send_zmq_message(topic_, state_data_sbuf_);
        }

        // 4. Re-publish robot_config periodically (ZMQ has no persistence)
        publish_config();
    }

    /**
     * @brief Publish robot_config if enough time has elapsed since the last send.
     *
     * On the first call the config is serialised from StateLogger and cached.
     * Subsequent calls simply re-send the cached buffer every
     * `CONFIG_REPUBLISH_INTERVAL_SEC` seconds.  This gives ZMQ PUB the same
     * "late-subscriber" semantics as ROS 2's `transient_local` QoS.
     *
     * Called once during init (from g1_deploy_onnx_ref.cpp) and then on every
     * control-loop tick from `publish()`.
     */
    void publish_config() override {
        // Lazy-init: serialise once, reuse forever.
        if (config_sbuf_cache_.size() == 0) {
            auto config_opt = state_logger_.GetConfig();
            if (!config_opt.has_value() || config_opt->empty()) {
                throw std::runtime_error("[ZMQ Output ERROR] Cannot publish config: StateLogger config is empty");
            }
            pack_robot_config(config_sbuf_cache_, *config_opt);
            std::cout << "[ZMQ Output] Robot config cached ("
                      << config_opt->size() << " fields, "
                      << config_sbuf_cache_.size() << " bytes)" << std::endl;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - config_last_publish_time_).count();
        if (elapsed >= CONFIG_REPUBLISH_INTERVAL_SEC) {
            send_zmq_message(robot_config_topic_, config_sbuf_cache_);
            config_last_publish_time_ = now;
        }
    }

private:
    zmq::context_t realtime_debug_context_;                ///< ZMQ context (1 I/O thread).
    std::unique_ptr<zmq::socket_t> realtime_debug_socket_; ///< ZMQ PUB socket.

    std::string topic_;              ///< User-provided topic name (e.g. "g1_debug") for combined state+viz.
    std::string robot_config_topic_; ///< Topic for robot config messages.

    msgpack::sbuffer state_data_sbuf_;  ///< Reused each tick; cleared in pack_combined_state().

    // -- Config re-publish (ZMQ equivalent of ROS 2 transient_local) --
    static constexpr double CONFIG_REPUBLISH_INTERVAL_SEC = 2.0;
    msgpack::sbuffer config_sbuf_cache_;  ///< Serialised config (populated on first publish_config()).
    std::chrono::steady_clock::time_point config_last_publish_time_;

    /// Non-blocking send of [topic][msgpack payload] over the PUB socket.
    void send_zmq_message(const std::string& topic, const msgpack::sbuffer& sbuf) {
        zmq::message_t msg(topic.size() + sbuf.size());
        memcpy(msg.data(), topic.c_str(), topic.size());
        memcpy(static_cast<char*>(msg.data()) + topic.size(), sbuf.data(), sbuf.size());
        realtime_debug_socket_->send(msg, zmq::send_flags::dontwait);
    }

    /**
     * @brief Serialise both state-logger and visualisation data into state_data_sbuf_.
     *
     * Combines into a single msgpack map:
     *   - State-logger fields (body_q, body_dq, last_action, etc. in MuJoCo order)
     *   - Visualisation fields from output_data_map_ (targets, VR data, measured poses)
     *
     * This avoids sending two overlapping messages per tick.
     */
    void pack_combined_state(const DataBuffer<HeadingState>& heading_state_buffer) {
        state_data_sbuf_.clear();

        if (state_logger_.size() == 0) {
            return;
        }

        std::vector<StateLogger::Entry> entries = state_logger_.GetLatest(1);
        const StateLogger::Entry& state = entries[0];
        msgpack::packer<msgpack::sbuffer> pk(&state_data_sbuf_);

        HeadingState heading_state;
        bool has_heading_state = false;
        auto heading_state_data = heading_state_buffer.GetDataWithTime().data;
        if (heading_state_data) {
            heading_state = *heading_state_data;
            has_heading_state = true;
        }

        // State-logger fields: 18 base + 2 optional heading
        // Visualisation fields: output_data_map_.size() (typically 11)
        int num_state_fields = has_heading_state ? 20 : 18;
        int num_viz_fields = static_cast<int>(output_data_map_.size());
        pk.pack_map(num_state_fields + num_viz_fields);

        // ---- State-logger fields ----

        pk.pack("control_loop_type");
        pk.pack("cpp");

        pk.pack("index");
        pk.pack(state.index);

        pk.pack("ros_timestamp");
        pk.pack(state.ros_timestamp);

        pk.pack("base_quat");
        pk.pack_array(4);
        for (const auto& val : state.base_quat) pk.pack(val);

        pk.pack("base_ang_vel");
        pk.pack_array(3);
        for (const auto& val : state.base_ang_vel) pk.pack(val);

        pk.pack("body_torso_quat");
        pk.pack_array(4);
        for (const auto& val : state.body_torso_quat) pk.pack(val);

        pk.pack("body_torso_ang_vel");
        pk.pack_array(3);
        for (const auto& val : state.body_torso_ang_vel) pk.pack(val);

        // body_q: IsaacLab -> MuJoCo order, add default-angle offset
        pk.pack("body_q");
        pk.pack_array(state.body_q.size());
        if (state.body_q.size() == 29) {
            std::array<double, 29> body_q_mujoco;
            for (size_t i = 0; i < 29; ++i)
                body_q_mujoco[i] = state.body_q[isaaclab_to_mujoco[i]] + default_angles[i];
            for (const auto& val : body_q_mujoco) pk.pack(val);
        } else {
            for (const auto& val : state.body_q) pk.pack(val);
        }

        // body_dq: IsaacLab -> MuJoCo order (no offset for velocities)
        pk.pack("body_dq");
        pk.pack_array(state.body_dq.size());
        if (state.body_dq.size() == 29) {
            std::array<double, 29> body_dq_mujoco;
            for (size_t i = 0; i < 29; ++i)
                body_dq_mujoco[i] = state.body_dq[isaaclab_to_mujoco[i]];
            for (const auto& val : body_dq_mujoco) pk.pack(val);
        } else {
            for (const auto& val : state.body_dq) pk.pack(val);
        }

        // last_action: IsaacLab -> MuJoCo order, scale + default-angle offset
        pk.pack("last_action");
        pk.pack_array(state.last_action.size());
        if (state.last_action.size() == 29) {
            std::array<double, 29> last_action_mujoco;
            for (size_t i = 0; i < 29; ++i)
                last_action_mujoco[i] = state.last_action[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i];
            for (const auto& val : last_action_mujoco) pk.pack(val);
        } else {
            for (const auto& val : state.last_action) pk.pack(val);
        }

        pk.pack("left_hand_q");
        pk.pack_array(state.left_hand_q.size());
        for (const auto& val : state.left_hand_q) pk.pack(val);

        pk.pack("left_hand_dq");
        pk.pack_array(state.left_hand_dq.size());
        for (const auto& val : state.left_hand_dq) pk.pack(val);

        pk.pack("right_hand_q");
        pk.pack_array(state.right_hand_q.size());
        for (const auto& val : state.right_hand_q) pk.pack(val);

        pk.pack("right_hand_dq");
        pk.pack_array(state.right_hand_dq.size());
        for (const auto& val : state.right_hand_dq) pk.pack(val);

        pk.pack("last_left_hand_action");
        pk.pack_array(state.last_left_hand_action.size());
        for (const auto& val : state.last_left_hand_action) pk.pack(val);

        pk.pack("last_right_hand_action");
        pk.pack_array(state.last_right_hand_action.size());
        for (const auto& val : state.last_right_hand_action) pk.pack(val);

        pk.pack("token_state");
        if (state.has_post_state_data && !state.token_state.empty()) {
            pk.pack_array(state.token_state.size());
            for (const auto& val : state.token_state) pk.pack(val);
        } else {
            pk.pack_array(0);
        }

        // Motor temperature: hardware order, 2 values per motor (winding, driver)
        pk.pack("motor_temperature");
        pk.pack_array(state.motor_temperature.size());
        for (const auto& val : state.motor_temperature) pk.pack(val);

        if (has_heading_state) {
            pk.pack("init_base_quat");
            pk.pack_array(4);
            for (const auto& val : heading_state.init_base_quat) pk.pack(val);

            pk.pack("delta_heading");
            pk.pack(heading_state.delta_heading);
        }

        // ---- Visualisation fields (from output_data_map_) ----
        // Adds: base_trans_target, base_quat_target, body_q_target,
        //       base_trans_measured, base_quat_measured, body_q_measured,
        //       left_hand_q_measured, right_hand_q_measured,
        //       vr_3point_position, vr_3point_orientation, vr_3point_compliance
        for (const auto& [key, values] : output_data_map_) {
            pk.pack(key);
            pk.pack_array(values.size());
            for (const auto& val : values) pk.pack(val);
        }
    }

    /// Serialise a config map into the given sbuffer.
    static void pack_robot_config(msgpack::sbuffer& sbuf,
                                  const std::map<std::string, std::variant<std::string, int, double, bool>>& config) {
        sbuf.clear();
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);
        pk.pack_map(config.size());
        for (const auto& [key, value] : config) {
            pk.pack(key);
            std::visit([&pk](auto&& arg) { pk.pack(arg); }, value);
        }
    }

};

#endif // ZMQ_OUTPUT_HANDLER_HPP
