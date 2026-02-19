/**
 * @file zmq_endpoint_interface.hpp
 * @brief ZMQ-based input interface for receiving streamed pose / motion data.
 *
 * ZMQEndpointInterface combines SimpleKeyboard-style local controls with
 * network-streamed motion data received via the ZMQ packed-message protocol.
 * Pressing **Enter** toggles between pre-loaded reference motions and live
 * ZMQ streaming.
 *
 * ## Keyboard Controls (when this interface is active)
 *
 *   Key    | Action
 *   -------|-------
 *   Enter  | Toggle ZMQ streaming on/off
 *   P/p    | Previous motion (non-streaming mode)
 *   N/n    | Next motion
 *   T/t    | Play / resume
 *   R/r    | Restart (frame 0, paused)
 *   ]      | Start control
 *   O/o    | Emergency stop
 *   Q/q    | Delta heading left
 *   E/e    | Delta heading right
 *   I/i    | Reinitialise heading
 *
 * ## Protocol Versions
 *
 * All versions carry `body_quat` and `frame_index` as required fields.
 * Additionally:
 *
 *   Version | Required                         | Optional
 *   --------|----------------------------------|---------------------------
 *   1       | joint_pos, joint_vel             | smpl_joints, smpl_pose
 *   2       | smpl_joints, smpl_pose           | joint_pos, joint_vel
 *   3       | joint_pos, joint_vel, smpl_joints, smpl_pose | —
 *
 * ## Optional Fields (all versions)
 *
 *   - `left_hand_joints`, `right_hand_joints` – 7-DOF Dex3 joint values.
 *   - `vr_position` (9 doubles) – enables VR 3-point tracking mode.
 *   - `vr_orientation` (12 doubles) – defaults used if absent.
 *   - `vr_compliance` (3 doubles) – **IGNORED** (compliance is keyboard-controlled).
 *   - `catch_up` (bool, default true) – controls gap tolerance for motion sync.
 *   - `heading_increment` (scalar) – incremental heading adjustment per message.
 *
 * ## Streaming Architecture
 *
 *   1. A background ZMQPackedMessageSubscriber thread receives messages and
 *      copies them into `buffered_header_` / `buffered_buffers_` under `data_mutex_`.
 *   2. update() reads keyboard input and resets per-frame flags.
 *   3. handle_input() checks `has_new_data_`, decodes the buffered message via
 *      DecodeIntoMotionSequence() (which delegates to StreamedMotionMerger for
 *      sliding-window logic), and swaps the current_motion pointer.
 */

#ifndef ZMQ_ENDPOINT_INTERFACE_HPP
#define ZMQ_ENDPOINT_INTERFACE_HPP

#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <limits>

#include "input_interface.hpp"
#include "zmq_packed_message_subscriber.hpp"
#include "streamed_motion_merger.hpp"

/**
 * @class ZMQEndpointInterface
 * @brief InputInterface that streams pose / motion data over ZMQ and merges
 *        it into a MotionSequence for real-time playback.
 *
 * Can operate standalone (keyboard + network) or as a delegate inside
 * InterfaceManager / GamepadManager / ZMQManager.
 */
class ZMQEndpointInterface : public InputInterface {
public:
    /// Compile-time toggle for debug log output.
    static constexpr bool DEBUG_LOGGING = true;
    
    // ------------------------------------------------------------------
    // Per-frame action flags (reset at the start of every update() call)
    // ------------------------------------------------------------------
    bool motion_prev = false;      ///< Previous pre-loaded motion.
    bool motion_next = false;      ///< Next pre-loaded motion.
    bool play_motion = false;      ///< Play / resume.
    bool motion_restart = false;   ///< Restart (frame 0, paused).
    bool start_control = false;    ///< Start control system.
    bool stop_control = false;     ///< Emergency stop.
    bool delta_left = false;       ///< Heading nudge left.
    bool delta_right = false;      ///< Heading nudge right.
    bool reinitialize = false;     ///< Recapture IMU heading.
    bool toggle_zmq_mode = false;  ///< Toggle ZMQ streaming on/off (Enter key).

    /// When true, handle_input() reads from the ZMQ stream instead of
    /// pre-loaded reference motions.
    bool use_zmq_stream = false;
    
    /// Reusable sliding-window merger that handles frame alignment, gap
    /// detection, and catch-up logic for streamed motion data.
    StreamedMotionMerger motion_merger_;
    
    /// Protocol version established by the first received ZMQ message.
    /// −1 = not yet established.  Changing mid-session is an error.
    int active_protocol_version_ = -1;
    
    /// Shared pointer to the latest merged motion sequence from ZMQ data.
    std::shared_ptr<MotionSequence> streamed_motion_;
    /// Global frame index corresponding to streamed_motion_[0].
    int stream_window_start_ = 0;

    static constexpr std::string_view LOCALHOST = "localhost";

    ZMQEndpointInterface(
        const std::string& host = std::string(LOCALHOST),
        int port = 5556,
        const std::string& topic = "pose",
        bool use_conflate = false,
        bool verbose = false
    ) : InputInterface(), host_(host), port_(port), topic_(topic), verbose_(verbose), is_localhost_(host == LOCALHOST) {
        type_ = InputType::NETWORK;
        
        // Set terminal to non-blocking mode (same as SimpleKeyboard)
        tcgetattr(STDIN_FILENO, &old_termios_);
        struct termios new_termios = old_termios_;
        new_termios.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
        
        // Create ZMQ subscriber
        subscriber_ = std::make_unique<ZMQPackedMessageSubscriber>(
            host, port, topic,
            /*timeout_ms=*/100,
            verbose,
            use_conflate,
            /*rcv_hwm=*/ use_conflate ? 1 : 3
        );
        
        // Setup callback to receive and buffer pose data
        subscriber_->SetOnDecodedMessage(
            [this](const std::string& topic,
                   const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
                   const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
                this->OnPoseDataReceived(topic, hdr, bufs);
            }
        );
        
        // Start background receiving thread
        subscriber_->Start();
        
        // Initialize streamed motion buffer (reserve large capacity for streaming)
        ResetStreamedMotion();
        
        std::cout << "[ZMQEndpointInterface] Connected to " << host << ":" << port 
                  << " topic='" << topic << "'" << std::endl;
        std::cout << "[ZMQEndpointInterface] Press ENTER to toggle between loaded motions and ZMQ stream" << std::endl;
    }
    
    ~ZMQEndpointInterface() {
        if (subscriber_) {
            subscriber_->Stop();
        }
        // Restore terminal
        tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
    }
    
    // Flag to trigger safety reset in handle_input
    bool trigger_safety_reset = false;

    // Update is called each frame - read keyboard and check for network data
    void update() override {
        // Check for safety reset trigger from manager
        if (CheckAndClearSafetyReset()) {
            use_zmq_stream = false;
            trigger_safety_reset = true;
            std::cout << "[ZMQEndpointInterface] Safety reset triggered: will disable ZMQ streaming and return to reference motion" << std::endl;
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
        toggle_zmq_mode = false;

        // Read keyboard input (same as SimpleKeyboard, but without planner keys)
        // Using shared buffered reading
        char ch;
        while (ReadStdinChar(ch)) {
            switch (ch) {
                case 'p':
                case 'P': motion_prev = true; break;
                case 'n':
                case 'N': motion_next = true; break;
                case 't':
                case 'T': play_motion = true; break;
                case 'r':
                case 'R': motion_restart = true; break;
                case ']': start_control = true; break;
                case 'o':
                case 'O': stop_control = true; break;
                case 'q':
                case 'Q': delta_left = true; break;
                case 'e':
                case 'E': delta_right = true; break;
                case 'i':
                case 'I': reinitialize = true; break;
                case '\n': toggle_zmq_mode = true; break; // Toggle ZMQ streaming
            }
        }

    }
    
    // Handle input and update motion data
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
                // Encoder mode will be read from the motion's encode_mode
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

            // Disable ZMQ streaming and return to reference motion
            use_zmq_stream = false;
            ResetStreamedMotion(); // Reset motion merger and protocol version
            
            std::cout << "Safety reset: ZMQ streaming disabled, returned to reference motion at frame 0" << std::endl;
        }

        // Handle ZMQ mode toggle
        if (toggle_zmq_mode) {
            use_zmq_stream = !use_zmq_stream;
            if (use_zmq_stream) {
                std::cout << "=====================================" << std::endl;
                std::cout << "ZMQ STREAMING MODE: ENABLED" << std::endl;
                std::cout << "=====================================" << std::endl;
                std::cout << "Using pose data from " << host_ << ":" << port_ << std::endl;
                std::cout << "Press ENTER again to return to loaded motions" << std::endl;
                // reset the heading state
                {
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    operator_state.play = false;
                    reinitialize_heading = true; // reset the heading state
                }
                // reset streaming buffers when enabling to avoid mixing with stale data
                ResetStreamedMotion(); // This also resets protocol version in the merger
                has_new_data_ = false;
            } else {
                std::cout << "=====================================" << std::endl;
                std::cout << "ZMQ STREAMING MODE: DISABLED" << std::endl;
                std::cout << "=====================================" << std::endl;
                std::cout << "Using pre-loaded motion data" << std::endl;
                
                // Encoder mode will be read from the motion's encode_mode
                
                // reset the current motion and frame
                {
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    operator_state.play = false;
                    reinitialize_heading = true;
                    current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_); // current motion is the pre-loaded motion
                    current_frame = 0; // current frame is 0
                    if (current_motion->GetEncodeMode() >= 0) {
                        current_motion->SetEncodeMode(0);
                    }
                }
                // reset the streamed motion (also resets protocol version)
                ResetStreamedMotion();
                has_new_data_ = false;
            }
        }
        if (stop_control) { operator_state.stop = true; }
        if (start_control) { operator_state.start = true; }

        // Handle delta heading controls
        if (delta_left) {
            auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
            HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
            double new_delta = current_state.delta_heading + 0.1;
            heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
            std::cout << "Delta heading left: " << new_delta << " rad" << std::endl;
        }

        if (delta_right) {
            auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
            HeadingState current_state = current_heading_state ? *current_heading_state : HeadingState();
            double new_delta = current_state.delta_heading - 0.1;
            heading_state_buffer.SetData(HeadingState(current_state.init_base_quat, new_delta));
            std::cout << "Delta heading right: " << new_delta << " rad" << std::endl;
        }

        // If ZMQ mode is active, use streamed motion data
        if (use_zmq_stream) {
            // Check and decode new network data if available
            std::shared_ptr<MotionSequence> new_motion;
            int frame_offset_adjustment = 0;
            bool did_catchup = false;
            int protocol_version_for_mode_update = -1;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                if (has_new_data_) {
                    has_new_data_ = false; // consumed
                    if constexpr (DEBUG_LOGGING) {
                        std::cout << "[ZMQEndpointInterface] *** Starting ZMQ processing ***" << std::endl;
                    }
                    // Decode into a new MotionSequence with current playback position
                    auto result = DecodeIntoMotionSequence(current_frame, streamed_motion_, stream_window_start_, heading_state_buffer);
                    
                    // Check if protocol version change was detected
                    if (!result.motion && result.protocol_version != 0) {
                        // Protocol version changed - decoder rejected it
                        std::cerr << "✗✗✗ ERROR: Protocol version changed from " << active_protocol_version_
                                  << " to " << result.protocol_version << " during active ZMQ session!" << std::endl;
                        std::cerr << "✗✗✗ This is not allowed. Exiting ZMQ streaming mode for safety." << std::endl;
                        
                        // Disable ZMQ streaming and reset
                        use_zmq_stream = false;
                        
                        // Encoder mode will be read from the reference motion's encode_mode
                        
                        // Reset to reference motion
                        {
                            std::lock_guard<std::mutex> lock(current_motion_mutex);
                            operator_state.play = false;
                            reinitialize_heading = true;
                            current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
                            current_frame = 0;
                            if (current_motion->GetEncodeMode() >= 0) {
                                current_motion->SetEncodeMode(0);
                            }
                        }
                        
                        std::cout << "=====================================" << std::endl;
                        std::cout << "ZMQ STREAMING MODE: FORCE DISABLED" << std::endl;
                        std::cout << "=====================================" << std::endl;
                        std::cout << "Returned to reference motion. Re-enable ZMQ mode to continue." << std::endl;
                        
                        // Skip processing this frame
                        return;
                    }
                    
                    if (result.motion) {
                        // Determine encode_mode based on protocol version (only once when first established)
                        // Version 1: Use encoder mode 0 (joint-based)
                        // Version 2/3: Use encoder mode 2 (SMPL-based)
                        if constexpr (DEBUG_LOGGING) {
                            std::cout << "[ZMQEndpointInterface] active_protocol_version_=" << active_protocol_version_ << std::endl;
                            std::cout << "[ZMQEndpointInterface] result.motion->GetEncodeMode()=" << result.motion->GetEncodeMode() << std::endl;
                        }
                    
                        
                        new_motion = result.motion;
                        std::cout << "[ZMQEndpointInterface] motion name: " << new_motion->name << std::endl;
                        stream_window_start_ = result.window_start;
                        frame_offset_adjustment = result.frame_offset_adjustment;
                        did_catchup = result.did_catchup_reset;
                        
                        if constexpr (DEBUG_LOGGING) {
                            int window_end_msg_idx = stream_window_start_ + result.frame_step * (new_motion->timesteps - 1);
                            std::cout << "[ZMQEndpointInterface] Merged streamed data: " 
                                      << new_motion->timesteps << " current-rate frames, "
                                      << "window [" << stream_window_start_ << ".." << window_end_msg_idx << "] (message-index)"
                                      << ", frame_step=" << result.frame_step
                                      << ", frame_offset_adjustment=" << frame_offset_adjustment
                                      << ", did_catchup=" << did_catchup << std::endl;
                        }
                    }
                    if constexpr (DEBUG_LOGGING) {
                        std::cout << "[ZMQEndpointInterface] *** End of ZMQ decoding processing ***" << std::endl;
                    }
                }
            }
            
            // update streamed_motion_ and current_frame if we have new data
            if (new_motion) {
                streamed_motion_ = new_motion;
                
                // Handle catch-up reset: when window was reset due to large gap, start from beginning
                if (did_catchup) {
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    current_frame = 0;
                    current_motion = streamed_motion_;  // Assign shared_ptr directly for thread safety
                    operator_state.play = true; // Auto-play when entering ZMQ mode
                    reinitialize_heading = true;
                    
                    if constexpr (DEBUG_LOGGING) {
                        std::cout << "[ZMQEndpointInterface] Catch-up: Reset to frame 0 at global frame " 
                                  << stream_window_start_ << std::endl;
                    }
                } else {
                    // Normal case: Adjust current_frame to maintain global playback position after window shift
                    // current_frame represents "the next frame to be read" (not yet consumed)
                    int adjusted_frame = current_frame - frame_offset_adjustment;
                    
                    // Validate the adjustment doesn't cause discontinuities due to clamping
                    if (adjusted_frame < 0) {
                        if constexpr (DEBUG_LOGGING) {
                            std::cout << "[ZMQEndpointInterface] WARNING: Window shifted past playback position. "
                                      << "Skipping from global frame " << (stream_window_start_ - frame_offset_adjustment + current_frame)
                                      << " to " << stream_window_start_ << std::endl;
                        }
                        adjusted_frame = 0; // Start from beginning of new window
                    } else if (adjusted_frame >= streamed_motion_->timesteps) {
                        if constexpr (DEBUG_LOGGING) {
                            std::cout << "[ZMQEndpointInterface] WARNING: Playback position beyond new window. "
                                      << "Clamping to last frame." << std::endl;
                        }
                        // Safety: ensure we don't set negative frame index if timesteps is 0
                        adjusted_frame = (streamed_motion_->timesteps > 0) ? (streamed_motion_->timesteps - 1) : 0;
                    }
                    
                    std::lock_guard<std::mutex> lock(current_motion_mutex);
                    current_frame = adjusted_frame;
                    current_motion = streamed_motion_;  // Assign shared_ptr directly for thread safety
                    operator_state.play = true; // Auto-play when entering ZMQ mode
                }
                
            }
            return; // Skip keyboard motion controls when in ZMQ mode
        }
        
        // Standard keyboard controls (same as SimpleKeyboard, without planner)
        if (motion_prev && !motion_reader.motions.empty()) {
            motion_reader.current_motion_index_ =
                (motion_reader.current_motion_index_ - 1 + motion_reader.motions.size()) % motion_reader.motions.size();
            std::string motion_name;
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
                current_frame = 0;
                motion_name = current_motion->name;
                reinitialize_heading = true;
            }
        }

        if (motion_next && !motion_reader.motions.empty()) {
            motion_reader.current_motion_index_ = (motion_reader.current_motion_index_ + 1) % motion_reader.motions.size();
            std::string motion_name;
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                current_motion = motion_reader.GetMotionShared(motion_reader.current_motion_index_);
                current_frame = 0;
                motion_name = current_motion->name;
                reinitialize_heading = true;
            }
        }

        if (play_motion) {
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

        if (motion_restart) {
            {
                std::lock_guard<std::mutex> lock(current_motion_mutex);
                operator_state.play = false;
                current_frame = 0;
                reinitialize_heading = true;
            }
            std::cout << "Reset motion " << motion_reader.current_motion_index_ << " to frame 0 (paused)" << std::endl;
        }

        // Handle reinitialize command
        if (reinitialize) {
            std::lock_guard<std::mutex> lock(current_motion_mutex);
            reinitialize_heading = true;
            std::cout << "Reinitialized base quaternion and reset delta heading to 0" << std::endl;
        }
    }

    // Public method to trigger ZMQ mode toggle (for programmatic control from GamepadManager)
    void TriggerZMQToggle() {
        toggle_zmq_mode = true;
    }

    std::optional<std::chrono::steady_clock::time_point> GetLastUpdateTime() const override {
      if (is_localhost_) {
        return data_timestamp_;
      }
      return last_receive_time_;
    }
    
private:
    /// Reset the streamed motion buffer, merger state, and protocol version.
    /// Called on construction, when toggling ZMQ mode, and on safety reset.
    void ResetStreamedMotion() {
        motion_merger_.Reset();
        active_protocol_version_ = -1;  // Reset protocol version tracking
        // Update legacy fields for backward compatibility
        streamed_motion_ = std::make_shared<MotionSequence>();
        streamed_motion_->name = "streamed";
        streamed_motion_->ReserveCapacity(15000, 29, 1, 1, 0, 0); // max 15k frames, 29 joints, 1 body, 1 quat
        stream_window_start_ = 0;
        data_timestamp_.reset();
        last_receive_time_.reset();
    }
    
    /// Outcome of DecodeIntoMotionSequence().
    struct DecodeResult {
        std::shared_ptr<MotionSequence> motion;  ///< Merged motion (nullptr on failure / version change).
        int window_start = 0;                     ///< Global frame index of motion[0].
        int frame_offset_adjustment = 0;          ///< Subtract from current_frame for window shift.
        bool did_catchup_reset = false;            ///< True → caller should reset playback to frame 0.
        int frame_step = 1;                        ///< Detected stride between frame indices.
        int protocol_version = 0;                  ///< Protocol version from the message (1, 2, or 3).
    };
    
    /**
     * @brief Decode buffered network data into a new MotionSequence.
     *
     * Called from handle_input() (with data_mutex_ held) whenever `has_new_data_`
     * is true.  This method:
     *  1. Parses the buffered JSON header to determine field indices and dtypes.
     *  2. Validates required fields for the detected protocol version.
     *  3. Decodes binary buffers into typed C++ containers (joint_pos, body_quat, …).
     *  4. Delegates to StreamedMotionMerger::MergeIncomingData() for sliding-window logic.
     *  5. Sets encoder mode on the resulting motion based on protocol version.
     *  6. Updates VR / hand-joint buffers if the corresponding optional fields are present.
     *
     * @param current_playback_frame  Current playback cursor in the old motion.
     * @param old_motion              Previous streamed motion (for window overlap).
     * @param old_window_start        Global frame index of old_motion[0].
     * @param heading_state_buffer    Heading buffer (for heading_increment field).
     * @return DecodeResult describing the merged motion and playback adjustments.
     */
    DecodeResult DecodeIntoMotionSequence(int current_playback_frame, 
                                          std::shared_ptr<MotionSequence> old_motion,
                                          int old_window_start,
                                          DataBuffer<HeadingState>& heading_state_buffer) {
        DecodeResult result;
        if (buffered_buffers_.empty()) {
            std::cerr << "[ZMQEndpointInterface] No buffered buffers" << std::endl;
            return result;
        }
        
        // Track timing between decode calls
        uint64_t decode_start_time = std::chrono::steady_clock::now().time_since_epoch().count() / 1000000; // milliseconds
        
        // Check protocol version
        int protocol_version = buffered_header_.version;
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ZMQEndpointInterface] Protocol version: " << protocol_version << std::endl;
        }
        
        // Find expected fields by name (including frame_index for alignment)
        int joint_pos_idx = -1, joint_vel_idx = -1, body_quat_idx = -1, frame_index_idx = -1, smpl_joints_idx = -1, smpl_pose_idx = -1;
        int left_hand_joints_idx = -1, right_hand_joints_idx = -1, catch_up_idx = -1;
        int heading_increment_idx = -1;
        int timestamp_monotonic_idx = -1;
        // VR 3-point tracking fields (optional)
        int vr_position_idx = -1, vr_orientation_idx = -1, vr_compliance_idx = -1;
        
        for (size_t i = 0; i < buffered_header_.fields.size(); ++i) {
            const auto& f = buffered_header_.fields[i];
            if (f.name == "joint_pos") joint_pos_idx = static_cast<int>(i);
            else if (f.name == "joint_vel") joint_vel_idx = static_cast<int>(i);
            else if (f.name == "body_quat_w" || f.name == "body_quat") body_quat_idx = static_cast<int>(i);
            else if (f.name == "frame_index" || f.name == "last_smpl_global_frames") frame_index_idx = static_cast<int>(i);
            else if (f.name == "smpl_joints") smpl_joints_idx = static_cast<int>(i);
            else if (f.name == "smpl_pose") smpl_pose_idx = static_cast<int>(i);
            else if (f.name == "left_hand_joints") left_hand_joints_idx = static_cast<int>(i);
            else if (f.name == "right_hand_joints") right_hand_joints_idx = static_cast<int>(i);
            else if (f.name == "catch_up") catch_up_idx = static_cast<int>(i);
            else if (f.name == "heading_increment") heading_increment_idx = static_cast<int>(i);
            else if (f.name == "timestamp_monotonic") timestamp_monotonic_idx = static_cast<int>(i);
            // VR 3-point tracking fields
            else if (f.name == "vr_position") vr_position_idx = static_cast<int>(i);
            else if (f.name == "vr_orientation") vr_orientation_idx = static_cast<int>(i);
            else if (f.name == "vr_compliance") vr_compliance_idx = static_cast<int>(i);
        }
        
        // Validate required fields based on protocol version
        // body_quat and frame_index are required for both versions
        if (body_quat_idx < 0) {
            std::cerr << "[ZMQEndpointInterface] Missing required field 'body_quat' (or 'body_quat_w')" << std::endl;
            return result;
        }
        
        if (frame_index_idx < 0) {
            std::cerr << "[ZMQEndpointInterface] Missing required field 'frame_index' (or 'last_smpl_global_frames')" << std::endl;
            return result;
        }
        
        if (protocol_version == 2 || protocol_version == 3) {
            // Version 2/3: require smpl_joints, smpl_pose (joint_pos/joint_vel optional for v2, required for v3)
            if (smpl_joints_idx < 0) {
                std::cerr << "[ZMQEndpointInterface] Version " << protocol_version
                          << " missing required field 'smpl_joints' " << std::endl;
                return result;
            }
            if (smpl_pose_idx < 0) {
                std::cerr << "[ZMQEndpointInterface] Version " << protocol_version
                          << " missing required field 'smpl_pose'" << std::endl;
                return result;
            }
            if (protocol_version == 3) {
                // Version 3 additionally requires joint_pos and joint_vel
                if (joint_pos_idx < 0 ) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 missing required field 'joint_pos'" << std::endl;
                    return result;
                }
                if (joint_vel_idx < 0) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 missing required field 'joint_vel'" << std::endl;
                    return result;
                }
            }
        } else if (protocol_version == 1) {
            // Version 1: requires joint_pos and joint_vel (smpl_joints optional)
            if (joint_pos_idx < 0 || joint_vel_idx < 0) {
                std::cerr << "[ZMQEndpointInterface] Version 1 missing required fields (joint_pos, joint_vel)" << std::endl;
                return result;
            }
        } else {
            std::cerr << "[ZMQEndpointInterface] Unsupported protocol version: " << protocol_version << std::endl;
            return result;
        }
        
        // Determine num_frames and num_joints from available fields
        int num_frames = 0;
        int num_joints = 0;
        
        // Get num_frames from the primary required field for each version
        if (protocol_version == 2 || protocol_version == 3) {
            // Version 2/3: Get num_frames from smpl_joints (required)
            const auto& smpl_field = buffered_header_.fields[smpl_joints_idx];
            if (smpl_field.shape.size() < 2) {
                std::cerr << "[ZMQEndpointInterface] Invalid smpl_joints shape" << std::endl;
                return result;
            }
            int num_frames_smpl = static_cast<int>(smpl_field.shape[0]);
            if (num_frames_smpl <= 0) {
                std::cerr << "[ZMQEndpointInterface] Invalid number of frames from smpl_joints: " << num_frames_smpl << std::endl;
                return result;
            }

            // For version 3, also validate that joint_pos has consistent frame count
            if (protocol_version == 3) {
                const auto& joint_pos_field = buffered_header_.fields[joint_pos_idx];
                if (joint_pos_field.shape.size() != 2) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 has invalid joint_pos shape (expected [N, num_joints])" << std::endl;
                    return result;
                }
                const auto& joint_vel_field = buffered_header_.fields[joint_vel_idx];
                if (joint_vel_field.shape.size() != 2) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 has invalid joint_vel shape (expected [N, num_joints])" << std::endl;
                    return result;
                }
                int num_frames_joint = static_cast<int>(joint_pos_field.shape[0]);
                if (num_frames_joint != num_frames_smpl) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 frame count mismatch between smpl_joints (" 
                              << num_frames_smpl << ") and joint_pos (" << num_frames_joint << ")" << std::endl;
                    return result;
                }
                int num_frames_joint_vel = static_cast<int>(joint_vel_field.shape[0]);
                if (num_frames_joint_vel != num_frames_smpl) {
                    std::cerr << "[ZMQEndpointInterface] Version 3 frame count mismatch between smpl_joints (" 
                              << num_frames_smpl << ") and joint_vel (" << num_frames_joint_vel << ")" << std::endl;
                    return result;
                }
            }
            num_frames = num_frames_smpl;
        } else if (protocol_version == 1) {
            // Version 1: Get num_frames from joint_pos (required)
            const auto& joint_pos_field = buffered_header_.fields[joint_pos_idx];
            if (joint_pos_field.shape.size() != 2) {
                std::cerr << "[ZMQEndpointInterface] Invalid joint_pos shape" << std::endl;
                return result;
            }
            const auto& joint_vel_field = buffered_header_.fields[joint_vel_idx];
            if (joint_vel_field.shape.size() != 2) {
                std::cerr << "[ZMQEndpointInterface] Invalid joint_vel shape" << std::endl;
                return result;
            }
            num_frames = static_cast<int>(joint_pos_field.shape[0]);
            if (num_frames != static_cast<int>(joint_vel_field.shape[0])) {
                std::cerr << "[ZMQEndpointInterface] Frame count mismatch between joint_pos and joint_vel" << std::endl;
                return result;
            }
        }
        
        if (num_frames <= 0) {
            std::cerr << "[ZMQEndpointInterface] Invalid number of frames: " << num_frames << std::endl;
            return result;
        }
        
        // Get num_joints if joint data is present
        if (joint_pos_idx >= 0 && joint_vel_idx >= 0) {
            const auto& joint_pos_field = buffered_header_.fields[joint_pos_idx];
            const auto& joint_vel_field = buffered_header_.fields[joint_vel_idx];
            
            // Validate shapes: expect [N, num_joints]
            if (joint_pos_field.shape.size() == 2 && joint_vel_field.shape.size() == 2) {
                num_joints = static_cast<int>(joint_pos_field.shape[1]);
                if (num_joints <= 0) {
                    std::cerr << "[ZMQEndpointInterface] Invalid number of joints: " << num_joints << std::endl;
                    return result;
                }
            }
        }
        
        bool needs_swap = buffered_header_.NeedsByteSwap();
        
        // ===== STEP 1: Decode all incoming data into temporary buffers =====
        
        // Decode joint positions and velocities if present
        std::vector<std::vector<double>> decoded_joint_pos;
        std::vector<std::vector<double>> decoded_joint_vel;
        bool has_joint_data = (joint_pos_idx >= 0 && joint_vel_idx >= 0 && num_joints > 0);
        
        if (has_joint_data) {
            // Decode joint positions
            decoded_joint_pos.resize(num_frames, std::vector<double>(num_joints));
            const auto& joint_pos_field = buffered_header_.fields[joint_pos_idx];
            const auto& pos_buf = buffered_buffers_[joint_pos_idx];
            if (joint_pos_field.dtype == "f32") {
                for (int frame = 0; frame < num_frames; ++frame) {
                    for (int joint = 0; joint < num_joints; ++joint) {
                        float val;
                        std::memcpy(&val, pos_buf.data() + (frame * num_joints + joint) * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        decoded_joint_pos[frame][joint] = static_cast<double>(val);
                    }
                }
            } else if (joint_pos_field.dtype == "f64") {
                for (int frame = 0; frame < num_frames; ++frame) {
                    for (int joint = 0; joint < num_joints; ++joint) {
                        double val;
                        std::memcpy(&val, pos_buf.data() + (frame * num_joints + joint) * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        decoded_joint_pos[frame][joint] = val;
                    }
                }
            }
            
            // Decode joint velocities
            decoded_joint_vel.resize(num_frames, std::vector<double>(num_joints));
            const auto& joint_vel_field = buffered_header_.fields[joint_vel_idx];
            const auto& vel_buf = buffered_buffers_[joint_vel_idx];
            if (joint_vel_field.dtype == "f32") {
                for (int frame = 0; frame < num_frames; ++frame) {
                    for (int joint = 0; joint < num_joints; ++joint) {
                        float val;
                        std::memcpy(&val, vel_buf.data() + (frame * num_joints + joint) * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        decoded_joint_vel[frame][joint] = static_cast<double>(val);
                    }
                }
            } else if (joint_vel_field.dtype == "f64") {
                for (int frame = 0; frame < num_frames; ++frame) {
                    for (int joint = 0; joint < num_joints; ++joint) {
                        double val;
                        std::memcpy(&val, vel_buf.data() + (frame * num_joints + joint) * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        decoded_joint_vel[frame][joint] = val;
                    }
                }
            }
        }
        
        // Decode body quaternions (required for both versions)
        // Support shapes: [N, num_quat_bodies, 4] or [N, 4] for single body
        const auto& quat_field = buffered_header_.fields[body_quat_idx];
        const auto& quat_buf = buffered_buffers_[body_quat_idx];
        
        // Determine number of quaternion bodies from shape
        int num_quat_bodies = 1;
        if (quat_field.shape.size() == 3) {
            num_quat_bodies = static_cast<int>(quat_field.shape[1]);
        } else if (quat_field.shape.size() == 2) {
            num_quat_bodies = 1;
        }
        
        // Decode quaternions: [frame][body][xyzw]
        std::vector<std::vector<std::array<double, 4>>> decoded_body_quat(num_frames);
        for (int frame = 0; frame < num_frames; ++frame) {
            decoded_body_quat[frame].resize(num_quat_bodies, {1.0, 0.0, 0.0, 0.0});
        }
        
        int quat_stride = num_quat_bodies * 4;
        
        if (quat_field.dtype == "f32") {
            for (int frame = 0; frame < num_frames; ++frame) {
                for (int body = 0; body < num_quat_bodies; ++body) {
                    for (int q = 0; q < 4; ++q) {
                        float val;
                        std::memcpy(&val, quat_buf.data() + (frame * quat_stride + body * 4 + q) * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        decoded_body_quat[frame][body][q] = static_cast<double>(val);
                    }
                }
            }
        } else if (quat_field.dtype == "f64") {
            for (int frame = 0; frame < num_frames; ++frame) {
                for (int body = 0; body < num_quat_bodies; ++body) {
                    for (int q = 0; q < 4; ++q) {
                        double val;
                        std::memcpy(&val, quat_buf.data() + (frame * quat_stride + body * 4 + q) * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        decoded_body_quat[frame][body][q] = val;
                    }
                }
            }
        }
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[ZMQEndpointInterface] Decoded body quaternions: " << num_quat_bodies << " bodies per frame" << std::endl;
        }
        
        // Decode SMPL joints if present
        // Expected shape: [N, num_smpl_joints, 3] or [N, 3] for single joint
        std::vector<std::vector<std::array<double, 3>>> decoded_smpl_joints; // [frame][joint][xyz]
        int num_smpl_joints = 0;
        bool has_smpl_joints = (smpl_joints_idx >= 0);
        
        if (has_smpl_joints) {
            const auto& smpl_field = buffered_header_.fields[smpl_joints_idx];
            const auto& smpl_buf = buffered_buffers_[smpl_joints_idx];
            
            // Determine shape: [N, num_smpl_joints, 3] or [N, 3]
            if (smpl_field.shape.size() == 3) {
                num_smpl_joints = static_cast<int>(smpl_field.shape[1]);
            } else if (smpl_field.shape.size() == 2) {
                num_smpl_joints = 1;
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid smpl_joints shape dimensions: " 
                          << smpl_field.shape.size() << std::endl;
                has_smpl_joints = false; // Invalid shape, skip decoding
            }
            
            if (has_smpl_joints && num_smpl_joints > 0) {
                decoded_smpl_joints.resize(num_frames);
                
                int stride = num_smpl_joints * 3;
                
                if (smpl_field.dtype == "f32") {
                    for (int frame = 0; frame < num_frames; ++frame) {
                        decoded_smpl_joints[frame].resize(num_smpl_joints);
                        for (int joint = 0; joint < num_smpl_joints; ++joint) {
                            for (int xyz = 0; xyz < 3; ++xyz) {
                                float val;
                                std::memcpy(&val, smpl_buf.data() + (frame * stride + joint * 3 + xyz) * sizeof(float), sizeof(float));
                                if (needs_swap) val = byte_swap(val);
                                decoded_smpl_joints[frame][joint][xyz] = static_cast<double>(val);
                            }
                        }
                    }
                } else if (smpl_field.dtype == "f64") {
                    for (int frame = 0; frame < num_frames; ++frame) {
                        decoded_smpl_joints[frame].resize(num_smpl_joints);
                        for (int joint = 0; joint < num_smpl_joints; ++joint) {
                            for (int xyz = 0; xyz < 3; ++xyz) {
                                double val;
                                std::memcpy(&val, smpl_buf.data() + (frame * stride + joint * 3 + xyz) * sizeof(double), sizeof(double));
                                if (needs_swap) val = byte_swap(val);
                                decoded_smpl_joints[frame][joint][xyz] = val;
                            }
                        }
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded smpl_joints: " << num_frames 
                              << " frames, " << num_smpl_joints << " joints" << std::endl;
                }
            }
        }
        
        // Decode SMPL poses if present
        // Expected shape: [N, num_poses, 3] or [N, 3] for single pose
        std::vector<std::vector<std::array<double, 3>>> decoded_smpl_pose; // [frame][pose][xyz]
        int num_smpl_poses = 0;
        bool has_smpl_pose = (smpl_pose_idx >= 0);
        
        if (has_smpl_pose) {
            const auto& smpl_pose_field = buffered_header_.fields[smpl_pose_idx];
            const auto& smpl_pose_buf = buffered_buffers_[smpl_pose_idx];
            
            // Determine shape: [N, num_poses, 3] or [N, 3]
            if (smpl_pose_field.shape.size() == 3) {
                num_smpl_poses = static_cast<int>(smpl_pose_field.shape[1]);
            } else if (smpl_pose_field.shape.size() == 2) {
                num_smpl_poses = 1;
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid smpl_pose shape dimensions: " 
                          << smpl_pose_field.shape.size() << std::endl;
                has_smpl_pose = false; // Invalid shape, skip decoding
            }
            
            if (has_smpl_pose && num_smpl_poses > 0) {
                decoded_smpl_pose.resize(num_frames);
                
                int stride = num_smpl_poses * 3;
                
                if (smpl_pose_field.dtype == "f32") {
                    for (int frame = 0; frame < num_frames; ++frame) {
                        decoded_smpl_pose[frame].resize(num_smpl_poses);
                        for (int pose = 0; pose < num_smpl_poses; ++pose) {
                            for (int xyz = 0; xyz < 3; ++xyz) {
                                float val;
                                std::memcpy(&val, smpl_pose_buf.data() + (frame * stride + pose * 3 + xyz) * sizeof(float), sizeof(float));
                                if (needs_swap) val = byte_swap(val);
                                decoded_smpl_pose[frame][pose][xyz] = static_cast<double>(val);
                            }
                        }
                    }
                } else if (smpl_pose_field.dtype == "f64") {
                    for (int frame = 0; frame < num_frames; ++frame) {
                        decoded_smpl_pose[frame].resize(num_smpl_poses);
                        for (int pose = 0; pose < num_smpl_poses; ++pose) {
                            for (int xyz = 0; xyz < 3; ++xyz) {
                                double val;
                                std::memcpy(&val, smpl_pose_buf.data() + (frame * stride + pose * 3 + xyz) * sizeof(double), sizeof(double));
                                if (needs_swap) val = byte_swap(val);
                                decoded_smpl_pose[frame][pose][xyz] = val;
                            }
                        }
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded smpl_pose: " << num_frames 
                              << " frames, " << num_smpl_poses << " poses" << std::endl;
                }
            }
        }
        
        // Decode hand joint positions if present (7 DOF joint values)
        bool has_left_hand_joints = (left_hand_joints_idx >= 0);
        bool has_right_hand_joints = (right_hand_joints_idx >= 0);
        auto [has_left_hand, left_hand_joint_values] = GetHandPose(true);
        auto [has_right_hand, right_hand_joint_values] = GetHandPose(false);
        
        if (has_left_hand_joints) {
            const auto& left_hand_field = buffered_header_.fields[left_hand_joints_idx];
            const auto& left_hand_buf = buffered_buffers_[left_hand_joints_idx];
            
            // Validate shape: expect [7] or [1, 7]
            int num_hand_joints = 0;
            if (left_hand_field.shape.size() == 1 && left_hand_field.shape[0] == 7) {
                num_hand_joints = 7;
            } else if (left_hand_field.shape.size() == 2 && left_hand_field.shape[1] == 7) {
                num_hand_joints = 7;
            }
            
            if (num_hand_joints == 7) {
                // Decode 7 joint values
                if (left_hand_field.dtype == "f32") {
                    for (int j = 0; j < 7; ++j) {
                        float val;
                        std::memcpy(&val, left_hand_buf.data() + j * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        left_hand_joint_values[j] = static_cast<double>(val);
                    }
                } else if (left_hand_field.dtype == "f64") {
                    for (int j = 0; j < 7; ++j) {
                        double val;
                        std::memcpy(&val, left_hand_buf.data() + j * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        left_hand_joint_values[j] = val;
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded left_hand_joints: [";
                    for (int j = 0; j < 7; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(4) << left_hand_joint_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid left_hand_joints shape" << std::endl;
                has_left_hand_joints = false;
            }
        }
        
        if (has_right_hand_joints) {
            const auto& right_hand_field = buffered_header_.fields[right_hand_joints_idx];
            const auto& right_hand_buf = buffered_buffers_[right_hand_joints_idx];
            
            // Validate shape: expect [7] or [1, 7]
            int num_hand_joints = 0;
            if (right_hand_field.shape.size() == 1 && right_hand_field.shape[0] == 7) {
                num_hand_joints = 7;
            } else if (right_hand_field.shape.size() == 2 && right_hand_field.shape[1] == 7) {
                num_hand_joints = 7;
            }
            
            if (num_hand_joints == 7) {
                // Decode 7 joint values
                if (right_hand_field.dtype == "f32") {
                    for (int j = 0; j < 7; ++j) {
                        float val;
                        std::memcpy(&val, right_hand_buf.data() + j * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        right_hand_joint_values[j] = static_cast<double>(val);
                    }
                } else if (right_hand_field.dtype == "f64") {
                    for (int j = 0; j < 7; ++j) {
                        double val;
                        std::memcpy(&val, right_hand_buf.data() + j * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        right_hand_joint_values[j] = val;
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded right_hand_joints: [";
                    for (int j = 0; j < 7; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(4) << right_hand_joint_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid right_hand_joints shape" << std::endl;
                has_right_hand_joints = false;
            }
        }
        
        // ===== Decode VR 3-point tracking data if present =====
        // VR 3-point format:
        //   vr_position: 9 doubles (left wrist xyz, right wrist xyz, head xyz) - REQUIRED for VR mode
        //   vr_orientation: 12 doubles (left quat wxyz, right quat wxyz, head quat wxyz) - optional
        //   vr_compliance: 3 doubles (left_arm, right_arm, head compliance) - optional
        bool has_vr_position = (vr_position_idx >= 0);
        bool has_vr_orientation = (vr_orientation_idx >= 0);
        bool has_vr_compliance = (vr_compliance_idx >= 0);
        
        // Default values for VR 3-point (from InputInterface defaults)
        std::array<double, 9> vr_position_values = {
            0.0903,  0.1615, -0.2411,   // left wrist xyz
            0.1280, -0.1522, -0.2461,   // right wrist xyz
            0.0241, -0.0081,  0.4028    // head xyz
        };
        std::array<double, 12> vr_orientation_values = {
            0.7295,  0.3145,  0.5533, -0.2506,   // left quat (w,x,y,z)
            0.7320, -0.2639,  0.5395,  0.3217,   // right quat (w,x,y,z)
            0.9991,  0.011,   0.0402, -0.0002    // head quat (w,x,y,z)
        };
        std::array<double, 3> vr_compliance_values = GetVR3PointCompliance();  // Use keyboard-controlled compliance
        
        if (has_vr_position) {
            const auto& vr_pos_field = buffered_header_.fields[vr_position_idx];
            const auto& vr_pos_buf = buffered_buffers_[vr_position_idx];
            
            // Validate shape: expect [9] or [1, 9] or [3, 3]
            size_t total_elements = 1;
            for (auto dim : vr_pos_field.shape) total_elements *= dim;
            
            if (total_elements == 9) {
                if (vr_pos_field.dtype == "f32") {
                    for (int j = 0; j < 9; ++j) {
                        float val;
                        std::memcpy(&val, vr_pos_buf.data() + j * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        vr_position_values[j] = static_cast<double>(val);
                    }
                } else if (vr_pos_field.dtype == "f64") {
                    for (int j = 0; j < 9; ++j) {
                        double val;
                        std::memcpy(&val, vr_pos_buf.data() + j * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        vr_position_values[j] = val;
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded vr_position: [";
                    for (int j = 0; j < 9; ++j) {
                        if (j > 0) std::cout << ", ";
                        if (j == 3 || j == 6) std::cout << " | ";
                        std::cout << std::fixed << std::setprecision(4) << vr_position_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid vr_position shape (expected 9 elements, got " 
                          << total_elements << ")" << std::endl;
                has_vr_position = false;
            }
        }
        
        if (has_vr_orientation) {
            const auto& vr_orient_field = buffered_header_.fields[vr_orientation_idx];
            const auto& vr_orient_buf = buffered_buffers_[vr_orientation_idx];
            
            // Validate shape: expect [12] or [1, 12] or [3, 4]
            size_t total_elements = 1;
            for (auto dim : vr_orient_field.shape) total_elements *= dim;
            
            if (total_elements == 12) {
                if (vr_orient_field.dtype == "f32") {
                    for (int j = 0; j < 12; ++j) {
                        float val;
                        std::memcpy(&val, vr_orient_buf.data() + j * sizeof(float), sizeof(float));
                        if (needs_swap) val = byte_swap(val);
                        vr_orientation_values[j] = static_cast<double>(val);
                    }
                } else if (vr_orient_field.dtype == "f64") {
                    for (int j = 0; j < 12; ++j) {
                        double val;
                        std::memcpy(&val, vr_orient_buf.data() + j * sizeof(double), sizeof(double));
                        if (needs_swap) val = byte_swap(val);
                        vr_orientation_values[j] = val;
                    }
                }
                
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Decoded vr_orientation: [";
                    for (int j = 0; j < 12; ++j) {
                        if (j > 0) std::cout << ", ";
                        if (j == 4 || j == 8) std::cout << " | ";
                        std::cout << std::fixed << std::setprecision(4) << vr_orientation_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "[ZMQEndpointInterface] Invalid vr_orientation shape (expected 12 elements, got " 
                          << total_elements << ")" << std::endl;
                has_vr_orientation = false;
            }
        }
        
        // Note: vr_compliance from ZMQ is intentionally IGNORED
        // We always use the keyboard-controlled compliance values (g/h/b/v keys)
        // This keeps compliance control consistent across all input modes
        if (has_vr_compliance) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] vr_compliance field present but IGNORED (using keyboard-controlled values instead)" << std::endl;
            }
        }
        
        // ===== STEP 2: Decode frame indices =====
        // Note: The merger will calculate frame_step and incoming_frame_start internally
        std::vector<int64_t> frame_indices;
        
        if (frame_index_idx >= 0) {
            const auto& frame_idx_field = buffered_header_.fields[frame_index_idx];
            const auto& frame_idx_buf = buffered_buffers_[frame_index_idx];
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] Raw message field '" << frame_idx_field.name 
                          << "' (dtype=" << frame_idx_field.dtype << ", size=" << frame_idx_buf.size() << " bytes)" << std::endl;
            }
            
            if (frame_idx_field.dtype == "i32") {
                int num_indices = frame_idx_buf.size() / sizeof(int32_t);
                frame_indices.resize(num_indices);
                
                for (int i = 0; i < num_indices; ++i) {
                    int32_t val;
                    std::memcpy(&val, frame_idx_buf.data() + i * sizeof(int32_t), sizeof(int32_t));
                    if (needs_swap) val = byte_swap(val);
                    frame_indices[i] = val;
                }
            } else if (frame_idx_field.dtype == "i64") {
                int num_indices = frame_idx_buf.size() / sizeof(int64_t);
                frame_indices.resize(num_indices);
                
                for (int i = 0; i < num_indices; ++i) {
                    int64_t val;
                    std::memcpy(&val, frame_idx_buf.data() + i * sizeof(int64_t), sizeof(int64_t));
                    if (needs_swap) val = byte_swap(val);
                    frame_indices[i] = val;
                }
            }
        }

        // Optional: decode heading_increment (single scalar, f32 or f64)
        if (heading_increment_idx >= 0) {
          double heading_increment = 0.0;
          const auto& dh_buf = buffered_buffers_[heading_increment_idx];
          const auto& dh_field = buffered_header_.fields[heading_increment_idx];
          if (dh_field.dtype == "f32") {
            float val = 0.0f;
            if (dh_buf.size() >= sizeof(float)) {
              std::memcpy(&val, dh_buf.data(), sizeof(float));
              if (needs_swap) val = byte_swap(val);
              heading_increment = static_cast<double>(val);
            }
          } else { // f64 or default
            double val = 0.0;
            if (dh_buf.size() >= sizeof(double)) {
              std::memcpy(&val, dh_buf.data(), sizeof(double));
              if (needs_swap) val = byte_swap(val);
              heading_increment = val;
            }
          }

          auto current_heading_state = heading_state_buffer.GetDataWithTime().data;
          HeadingState current_state =
            current_heading_state ? *current_heading_state : HeadingState();

          // Add increment to current heading
          heading_state_buffer.SetData(
              HeadingState(
                current_state.init_base_quat,
                current_state.delta_heading + heading_increment));
        }

        // Optional: decode monotonic timestamp (single scalar, f64)
        if (timestamp_monotonic_idx >= 0) {
          double timestamp_monotonic = 0.0;
          const auto& ts_buf = buffered_buffers_[timestamp_monotonic_idx];
          const auto& ts_field = buffered_header_.fields[timestamp_monotonic_idx];
          if (ts_field.dtype == "f64") {
            double val = 0.0;
            if (ts_buf.size() >= sizeof(double)) {
              std::memcpy(&val, ts_buf.data(), sizeof(double));
              if (needs_swap) val = byte_swap(val);
              timestamp_monotonic = val;
            }
          }
          if (is_localhost_)
          {
            auto duration_monotonic = std::chrono::duration<double>(timestamp_monotonic);
            auto time_point_monotonic = std::chrono::steady_clock::time_point(
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(duration_monotonic));
            data_timestamp_ = time_point_monotonic;
          }
        }

        // ===== Decode catch_up field if present =====
        // Default: catch_up = true (use MAX_GAP_FRAMES)
        // If catch_up = false: allow infinite delays (set max_gap_frames to very large value)
        bool catch_up_enabled = true; // Default to true if field not present
        if (catch_up_idx >= 0) {
            const auto& catch_up_field = buffered_header_.fields[catch_up_idx];
            const auto& catch_up_buf = buffered_buffers_[catch_up_idx];
            
            // Decode boolean value (support bool, i32, i64, u8)
            if (catch_up_field.dtype == "bool" || catch_up_field.dtype == "u8") {
                uint8_t val = 0;
                if (catch_up_buf.size() >= sizeof(uint8_t)) {
                    std::memcpy(&val, catch_up_buf.data(), sizeof(uint8_t));
                    catch_up_enabled = (val != 0);
                }
            } else if (catch_up_field.dtype == "i32") {
                int32_t val = 0;
                if (catch_up_buf.size() >= sizeof(int32_t)) {
                    std::memcpy(&val, catch_up_buf.data(), sizeof(int32_t));
                    if (needs_swap) val = byte_swap(val);
                    catch_up_enabled = (val != 0);
                }
            } else if (catch_up_field.dtype == "i64") {
                int64_t val = 0;
                if (catch_up_buf.size() >= sizeof(int64_t)) {
                    std::memcpy(&val, catch_up_buf.data(), sizeof(int64_t));
                    if (needs_swap) val = byte_swap(val);
                    catch_up_enabled = (val != 0);
                }
            }
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] catch_up field: " << (catch_up_enabled ? "true" : "false") << std::endl;
            }
        } else {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] catch_up field not present, using default: true" << std::endl;
            }
        }
        
        // ===== DEBUG: Print merged frame indices and decoded data =====
        if constexpr (DEBUG_LOGGING) {
            // Print first 20 frames (or all frames if fewer than 20)
            int print_frames = std::min(20, num_frames);
            
            std::cout << "[ZMQEndpointInterface] Decoded data (Version " << protocol_version << ", first " << print_frames << " frames";
            if (has_joint_data) std::cout << ", " << num_joints << " joints";
            if (has_smpl_joints) std::cout << ", " << num_smpl_joints << " smpl_joints";
            if (has_smpl_pose) std::cout << ", " << num_smpl_poses << " smpl_pose";
            std::cout << "):" << std::endl;
            
            for (int frame = 0; frame < print_frames; ++frame) {
                // Print frame index if available
                std::cout << "  Frame[" << frame << "]";
                if (frame < static_cast<int>(frame_indices.size())) {
                    std::cout << " (idx=" << frame_indices[frame] << ")";
                }
                
                // Print joint_pos and joint_vel if present
                if (has_joint_data && !decoded_joint_pos.empty() && !decoded_joint_vel.empty()) {
                    int print_joints = std::min(2, num_joints);
                    std::cout << " joint_pos: [";
                    for (int j = 0; j < print_joints; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(6) << decoded_joint_pos[frame][j];
                    }
                    std::cout << "], joint_vel: [";
                    for (int j = 0; j < print_joints; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(6) << decoded_joint_vel[frame][j];
                    }
                    std::cout << "]";
                }
                
                // Print body_quat (always present)
                std::cout << ", body_quat: [";
                int print_quat_bodies = std::min(2, static_cast<int>(decoded_body_quat[frame].size()));
                for (int b = 0; b < print_quat_bodies; ++b) {
                    if (b > 0) std::cout << "; ";
                    std::cout << "(";
                    for (int q = 0; q < 4; ++q) {
                        if (q > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(6) << decoded_body_quat[frame][b][q];
                    }
                    std::cout << ")";
                }
                std::cout << "]";
                
                // Print smpl_joints if present
                if (has_smpl_joints && frame < static_cast<int>(decoded_smpl_joints.size())) {
                    std::cout << ", smpl_joints: [";
                    int print_bodies = std::min(1, static_cast<int>(decoded_smpl_joints[frame].size()));
                    for (int b = 0; b < print_bodies; ++b) {
                        if (b > 0) std::cout << "; ";
                        std::cout << "(";
                        for (int xyz = 0; xyz < 3; ++xyz) {
                            if (xyz > 0) std::cout << ", ";
                            std::cout << std::fixed << std::setprecision(6) << decoded_smpl_joints[frame][b][xyz];
                        }
                        std::cout << ")";
                    }
                    std::cout << "]";
                }
                
                // Print smpl_pose if present
                if (has_smpl_pose && frame < static_cast<int>(decoded_smpl_pose.size())) {
                    std::cout << ", smpl_pose: [";
                    int print_poses = std::min(1, static_cast<int>(decoded_smpl_pose[frame].size()));
                    for (int p = 0; p < print_poses; ++p) {
                        if (p > 0) std::cout << "; ";
                        std::cout << "(";
                        for (int xyz = 0; xyz < 3; ++xyz) {
                            if (xyz > 0) std::cout << ", ";
                            std::cout << std::fixed << std::setprecision(6) << decoded_smpl_pose[frame][p][xyz];
                        }
                        std::cout << ")";
                    }
                    std::cout << "]";
                }
                
                std::cout << std::endl;
            }
        }
        
        // ===== STEP 3: Validate protocol version (application-specific) =====
        
        // Check protocol version before merging
        if (active_protocol_version_ == -1) {
            // First message - establish protocol version
            active_protocol_version_ = protocol_version;
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] Protocol version " << active_protocol_version_ << " established" << std::endl;
            }
        } else if (active_protocol_version_ != protocol_version) {
            // Protocol version changed - this is an error
            std::cerr << "[ZMQEndpointInterface] ERROR: Protocol version changed from " 
                      << active_protocol_version_ << " to " << protocol_version << std::endl;
            result.protocol_version = protocol_version;  // Signal the change to caller
            return result;
        }
        
        // ===== STEP 4: Package decoded data and call StreamedMotionMerger =====
        
        // Prepare IncomingData structure for the merger
        StreamedMotionMerger::IncomingData incoming_data;
        incoming_data.joint_pos = std::move(decoded_joint_pos);
        incoming_data.joint_vel = std::move(decoded_joint_vel);
        incoming_data.body_quat = std::move(decoded_body_quat);
        incoming_data.smpl_joints = std::move(decoded_smpl_joints);
        incoming_data.smpl_pose = std::move(decoded_smpl_pose);
        incoming_data.frame_indices = std::move(frame_indices);
        incoming_data.protocol_version = protocol_version;
        incoming_data.catch_up_enabled = catch_up_enabled;
        incoming_data.num_frames = num_frames;
        incoming_data.num_joints = num_joints;
        incoming_data.num_quat_bodies = num_quat_bodies;
        incoming_data.num_smpl_joints = num_smpl_joints;
        incoming_data.num_smpl_poses = num_smpl_poses;
        
        // Call the reusable merger to handle sliding window logic
        auto merge_result = motion_merger_.MergeIncomingData(incoming_data, current_playback_frame);
        
        // Check for merge failure
        if (!merge_result.motion) {
            std::cerr << "[ZMQEndpointInterface] Failed to merge incoming data" << std::endl;
            return result;
        }
        
        // Convert MergeResult to DecodeResult
        if (active_protocol_version_ == 1) {
            merge_result.motion->SetEncodeMode(0);  // Protocol 1: joint-based
        } else if (active_protocol_version_ == 2 || active_protocol_version_ == 3) {
            // Protocol versions 2 and 3 both use encoder mode 2 (SMPL-based)
            merge_result.motion->SetEncodeMode(2);
        }
        result.motion = merge_result.motion;
        result.window_start = merge_result.window_start;
        result.frame_offset_adjustment = merge_result.frame_offset_adjustment;
        result.did_catchup_reset = merge_result.did_catchup_reset;
        result.frame_step = merge_result.frame_step;
        result.protocol_version = merge_result.protocol_version;
        
        // Handle hand joints: set hand joint values directly from decoded data
        if (has_left_hand_joints || has_right_hand_joints) {
            has_hand_joints_ = true;
            
            if (has_left_hand_joints) {
                left_hand_joint_.SetData(left_hand_joint_values);
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Left hand joints set: [";
                    for (int j = 0; j < 7; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(4) << left_hand_joint_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            }
            
            if (has_right_hand_joints) {
                right_hand_joint_.SetData(right_hand_joint_values);
                if constexpr (DEBUG_LOGGING) {
                    std::cout << "[ZMQEndpointInterface] Right hand joints set: [";
                    for (int j = 0; j < 7; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(4) << right_hand_joint_values[j];
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }
        
        // Handle VR 3-point tracking: set buffers when vr_position is present
        // vr_position is required to enable VR mode; orientation uses default if not provided
        // compliance is ALWAYS from keyboard-controlled values (ignoring ZMQ data)
        if (has_vr_position) {
            vr_3point_position_.SetData(vr_position_values);
            vr_3point_orientation_.SetData(vr_orientation_values);
            if (has_vr_compliance) SetVR3PointCompliance(vr_compliance_values);
            has_vr_3point_control_ = true;
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[ZMQEndpointInterface] VR 3-point tracking ENABLED:" << std::endl;
                std::cout << "  Position [L|R|H]: [";
                for (int j = 0; j < 9; ++j) {
                    if (j > 0) std::cout << ", ";
                    if (j == 3 || j == 6) std::cout << "| ";
                    std::cout << std::fixed << std::setprecision(4) << vr_position_values[j];
                }
                std::cout << "]" << std::endl;
                std::cout << "  Orientation [L|R|H]: [";
                for (int j = 0; j < 12; ++j) {
                    if (j > 0) std::cout << ", ";
                    if (j == 4 || j == 8) std::cout << "| ";
                    std::cout << std::fixed << std::setprecision(4) << vr_orientation_values[j];
                }
                std::cout << "]" << (has_vr_orientation ? "" : " (default)") << std::endl;
                if (has_vr_compliance) {
                    std::cout << "  Compliance [L,R,H]: [";
                    for (int j = 0; j < 3; ++j) {
                        if (j > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(2) << vr_compliance_values[j];
                    }
                    std::cout << "] (keyboard-controlled)" << std::endl;
                }
            }
        }

        // log the decode interval and decode time
        uint64_t decode_end_time = std::chrono::steady_clock::now().time_since_epoch().count() / 1000000; // milliseconds
        if constexpr (DEBUG_LOGGING) {
            if (last_decode_time_ > 0) {
                uint64_t decode_time = decode_end_time - decode_start_time;
                uint64_t time_delta = decode_end_time - last_decode_time_;
                std::cout << "[ZMQEndpointInterface] Decode interval: " << time_delta << " ms, decode time: " << decode_time << " ms" << std::endl;
            }
        }
        last_decode_time_ = decode_end_time;

        return result;
    }
    
    /**
     * @brief ZMQ subscriber callback – invoked on the **background thread**.
     *
     * Copies the received header and buffer data into `buffered_header_` /
     * `buffered_buffers_` under `data_mutex_` and sets `has_new_data_` = true.
     * The actual decoding happens later on the main thread in handle_input().
     */
    void OnPoseDataReceived(
        const std::string& topic,
        const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
        const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // Buffer the received data for processing in handle_input (main thread)
        buffered_header_ = hdr;
        buffered_buffers_.clear();
        for (const auto& buf : bufs) {
            // Copy buffer data (BufferView is only valid during callback)
            std::vector<uint8_t> copied(static_cast<const uint8_t*>(buf.data),
                                       static_cast<const uint8_t*>(buf.data) + buf.size);
            buffered_buffers_.push_back(std::move(copied));
        }
        
        has_new_data_ = true;
        last_receive_time_ = std::chrono::steady_clock::now();
        receive_count_++;
    }
    
    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------
    std::string host_;    ///< ZMQ server hostname.
    int port_;            ///< ZMQ server port.
    std::string topic_;   ///< ZMQ subscription topic.
    bool verbose_;        ///< Verbose logging flag.
    
    /// Background subscriber for the pose / motion topic.
    std::unique_ptr<ZMQPackedMessageSubscriber> subscriber_;
    
    struct termios old_termios_;  ///< Saved terminal state for restoration on destruction.
    
    // ------------------------------------------------------------------
    // Thread-safe data buffering (written by ZMQ subscriber thread, read by input thread)
    // ------------------------------------------------------------------
    mutable std::mutex data_mutex_;           ///< Guards the fields below.
    bool has_new_data_ = false;               ///< True when a new message is waiting to be decoded.
    ZMQPackedMessageSubscriber::DecodedHeader buffered_header_;  ///< Latest JSON header.
    std::vector<std::vector<uint8_t>> buffered_buffers_;         ///< Copied binary field data.
    
    // ------------------------------------------------------------------
    // Timing / diagnostics
    // ------------------------------------------------------------------
    bool is_localhost_ = true;         ///< True if host_ is localhost (for directly comparing timestamps)
    std::optional<std::chrono::steady_clock::time_point> data_timestamp_{};  ///< Timestamp of last received message from XR source
    std::optional<std::chrono::steady_clock::time_point> last_receive_time_{}; ///< Timestamp of last OnPoseDataReceived (ms, monotonic).
    uint64_t receive_count_ = 0;       ///< Total number of messages received.
    uint64_t last_decode_time_ = 0;    ///< Timestamp of last DecodeIntoMotionSequence call (ms).
    
};

#endif // ZMQ_ENDPOINT_INTERFACE_HPP
