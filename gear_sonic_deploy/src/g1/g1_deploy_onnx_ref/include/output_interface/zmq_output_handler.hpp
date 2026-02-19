/**
 * @file zmq_output_handler.hpp
 * @brief ZMQ PUB output handler for publishing robot state data.
 *
 * ZMQOutputHandler sends a msgpack-serialised state map (built by the base
 * class's `pack_output_data_map()`) over a ZeroMQ PUB socket each control-
 * loop tick.
 *
 * ## Wire Format
 *
 * Each published ZMQ message is a single-part message:
 *
 *   [topic_prefix][msgpack payload]
 *
 * The topic prefix is a plain string (e.g. "state") prepended to the message
 * so that subscribers using `zmq::sockopt::subscribe` can filter by topic.
 * The msgpack payload is a `map<string, vector<double>>` containing target /
 * measured joint positions, base pose, VR data, etc.
 *
 * ## Socket Options
 *
 * - Send HWM = 10 (old messages are dropped rather than queued).
 * - Send buffer = 32 KB (small, predictable memory footprint).
 * - Linger = 0 (socket closes immediately without waiting for pending sends).
 * - Non-blocking send (dontwait) to avoid stalling the control loop.
 */

#ifndef ZMQ_OUTPUT_HANDLER_HPP
#define ZMQ_OUTPUT_HANDLER_HPP

#include <memory>
#include <iostream>
#include <zmq.hpp>
#include <msgpack.hpp>

#include "output_interface.hpp"
#include "../policy_parameters.hpp"  // For isaaclab_to_mujoco, default_angles, g1_action_scale

/**
 * @class ZMQOutputHandler
 * @brief OutputInterface that publishes state data over a ZMQ PUB socket.
 */
class ZMQOutputHandler : public OutputInterface {
public:

    /**
     * @brief Construct the handler: create a ZMQ PUB socket and bind to the given port.
     * @param logger  Reference to the shared StateLogger.
     * @param port    TCP port to bind the PUB socket to (e.g. 5557).
     * @param topic   Topic prefix prepended to each published message.
     */
    explicit ZMQOutputHandler(StateLogger& logger, int port, const std::string& topic) 
        : OutputInterface(logger), realtime_debug_context_(1), topic_(topic) {

        std::cout << "Initializing realtime debug socket" << std::endl;
        std::cout << "Binding to port: " << port << " and topic: " << topic_ << std::endl;
        realtime_debug_socket_ = std::make_unique<zmq::socket_t>(realtime_debug_context_, ZMQ_PUB);

        realtime_debug_socket_->set(zmq::sockopt::sndhwm, 10);     // Drop old messages quickly
        realtime_debug_socket_->set(zmq::sockopt::sndbuf, 32768);   // 32 KB send buffer
        realtime_debug_socket_->set(zmq::sockopt::linger, 0);       // No lingering on close
        realtime_debug_socket_->bind("tcp://*:" + std::to_string(port));

        // Pre-allocate the ZMQ message buffer (resized each tick in publish())
        realtime_debug_msg_ = std::make_unique<zmq::message_t>(8 + (3 + 4 + G1_NUM_MOTOR) * sizeof(double));
        
        std::cout << "[INFO] Realtime debug socket bound to port: " << port << std::endl;
        
        type_ = OutputType::ZMQ;
    }

    /**
     * @brief Pack state data and send it over the ZMQ PUB socket (non-blocking).
     *
     * Calls `pack_output_data_map()` from the base class, then prepends the
     * topic string and sends the combined buffer with `zmq::send_flags::dontwait`.
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

        pack_output_data_map(
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

        // Non blocking send - should have predictable timing in the order of microseconds
        // so no need to do this on a separate thread
        realtime_debug_msg_->rebuild(topic_.size() + output_data_sbuf_.size());

        // write topic:
        strcpy((char *)realtime_debug_msg_->data(), topic_.c_str());

        // write data:
        memcpy((char*)realtime_debug_msg_->data() + topic_.size(), output_data_sbuf_.data(), output_data_sbuf_.size());

        realtime_debug_socket_->send(*realtime_debug_msg_, zmq::send_flags::dontwait);
    }

private:
    zmq::context_t realtime_debug_context_;                ///< ZMQ context (1 I/O thread).
    std::unique_ptr<zmq::socket_t> realtime_debug_socket_; ///< ZMQ PUB socket.

    /// Pre-allocated ZMQ message buffer (rebuilt each tick to fit the payload).
    std::unique_ptr<zmq::message_t> realtime_debug_msg_;

    std::string topic_;  ///< Topic prefix string prepended to each outgoing message.
};

#endif // ZMQ_OUTPUT_HANDLER_HPP

