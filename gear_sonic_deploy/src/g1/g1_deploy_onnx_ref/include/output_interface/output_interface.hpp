/**
 * @file output_interface.hpp
 * @brief Abstract base class for all output / publishing handlers.
 *
 * OutputInterface defines the polymorphic contract for publishing robot state
 * and target data to external consumers (visualisation tools, logging systems,
 * Python training scripts, etc.).
 *
 * Concrete implementations:
 *   - ZMQOutputHandler  – publishes a msgpack-serialised state map over ZMQ PUB.
 *   - ROS2OutputHandler – publishes state and configuration over ROS 2 topics.
 *
 * ## Data Flow
 *
 * The control loop calls `publish()` once per tick on each registered output
 * interface.  The base class provides a shared helper `pack_output_data_map()`
 * that:
 *  1. Reads the latest entry from the StateLogger.
 *  2. Computes **target** joint positions and body pose (base_trans_target,
 *     base_quat_target, body_q_target) from the current motion frame, applying
 *     heading corrections.
 *  3. Copies **measured** joint positions and body orientation from the robot
 *     state.
 *  4. Rotates VR 3-point positions into the target body frame.
 *  5. Packs everything into a `std::map<string, vector<double>>` and
 *     serialises it with msgpack into `output_data_sbuf_`.
 *
 * Sub-classes then transmit `output_data_sbuf_` over their respective
 * transport (ZMQ, ROS 2, file, etc.).
 *
 * ## Thread Safety
 *
 * `publish()` is called from the control thread.  Implementations should not
 * block for extended periods to avoid jitter in the control loop.
 */

#ifndef OUTPUT_INTERFACE_HPP
#define OUTPUT_INTERFACE_HPP

#include <memory>
#include <array>
#include <vector>
#include <msgpack.hpp>

#include "../state_logger.hpp"
#include "../motion_data_reader.hpp"

/**
 * @class OutputInterface
 * @brief Abstract base class for publishing robot state / target data.
 *
 * Owns a reference to the StateLogger and provides `pack_output_data_map()`
 * for sub-classes to build the canonical output payload.
 */
class OutputInterface {
public:
    /// Identifies the concrete output transport.
    enum class OutputType {
        ROS2,      ///< ROS 2 DDS topics.
        NETWORK,   ///< Generic network (reserved).
        FILE,      ///< File-based logging (reserved).
        ZMQ,       ///< ZeroMQ PUB socket.
        UNKNOWN    ///< Default / uninitialised.
    };

    /// Virtual destructor for correct polymorphic cleanup.
    virtual ~OutputInterface() = default;

    /**
     * @brief Publish the current robot state and target data.
     *
     * Called once per control-loop tick for each registered output interface.
     *
     * @param vr_3point_position         VR 3-point positions [left_wrist, right_wrist, head] × xyz.
     * @param vr_3point_orientation      VR 3-point orientations [left, right, head] × quaternion wxyz.
     * @param vr_3point_compliance       VR compliance values [left_arm, right_arm, head].
     * @param left_hand_joint            Left-hand Dex3 joint positions (7 DOF).
     * @param right_hand_joint           Right-hand Dex3 joint positions (7 DOF).
     * @param init_ref_data_root_rot_array  Initial reference-data root rotation (quaternion wxyz).
     * @param heading_state_buffer       Thread-safe heading buffer (init quat + delta heading).
     * @param current_motion             Currently-active MotionSequence.
     * @param current_frame              Playback cursor within current_motion.
     */
    virtual void publish(
        const std::array<double, 9>& vr_3point_position,
        const std::array<double, 12>& vr_3point_orientation,
        const std::array<double, 3>& vr_3point_compliance,
        const std::array<double, 7>& left_hand_joint,
        const std::array<double, 7>& right_hand_joint,
        const std::array<double, 4>& init_ref_data_root_rot_array,
        DataBuffer<HeadingState>& heading_state_buffer,
        std::shared_ptr<const MotionSequence> current_motion,
        int current_frame
    ) = 0;

    /// @return The OutputType tag for this concrete implementation.
    virtual OutputType GetType() const {
        return type_;
    }

protected:

    /**
     * @brief Build the canonical output payload from state + motion data.
     *
     * Populates `output_data_map_` with the following keys and serialises the
     * result into `output_data_sbuf_` via msgpack:
     *
     *   Key                    | Size | Description
     *   -----------------------|------|------------
     *   base_trans_target      |  3   | Target base translation (heading-corrected).
     *   base_quat_target       |  4   | Target base quaternion (heading-corrected).
     *   body_q_target          | 29   | Target joint positions (MuJoCo order).
     *   base_trans_measured    |  3   | Measured base translation (fixed default).
     *   base_quat_measured     |  4   | Measured base quaternion from IMU.
     *   body_q_measured        | 29   | Measured joint positions (MuJoCo order + default offsets).
     *   left_hand_q_measured   |  7   | Left-hand Dex3 joint positions.
     *   right_hand_q_measured  |  7   | Right-hand Dex3 joint positions.
     *   vr_3point_position     |  9   | VR positions rotated into target body frame.
     *   vr_3point_orientation  | 12   | VR orientations (passed through).
     *   vr_3point_compliance   |  3   | VR compliance values (passed through).
     *
     * Joint ordering uses `isaaclab_to_mujoco` remapping and `default_angles`
     * offsets from policy_parameters.hpp.
     */
    void pack_output_data_map(
        const std::array<double, 9>& vr_3point_position,
        const std::array<double, 12>& vr_3point_orientation,
        const std::array<double, 3>& vr_3point_compliance,
        const std::array<double, 7>& left_hand_joint,
        const std::array<double, 7>& right_hand_joint,
        const std::array<double, 4>& init_ref_data_root_rot_array,
        DataBuffer<HeadingState>& heading_state_buffer,
        std::shared_ptr<const MotionSequence> current_motion,
        int current_frame
    )
    {
        // Static key strings (avoid repeated allocations)
        static const std::string kBaseTransTarget = "base_trans_target";
        static const std::string kBaseQuatTarget = "base_quat_target";
        static const std::string kBodyQTarget = "body_q_target";
        static const std::string kBaseTransMeasured = "base_trans_measured";
        static const std::string kBaseQuatMeasured = "base_quat_measured";
        static const std::string kBodyQMeasured = "body_q_measured";
        static const std::string kLeftHandQMeasured = "left_hand_q_measured";
        static const std::string kRightHandQMeasured = "right_hand_q_measured";
        static const std::string kVr3pointPosition = "vr_3point_position";
        static const std::string kVr3pointOrientation = "vr_3point_orientation";
        static const std::string kVr3pointCompliance = "vr_3point_compliance";

        std::vector<StateLogger::Entry> entries = state_logger_.GetLatest(1);
        const StateLogger::Entry& state = entries[0];

        // ---- Validate current motion frame ----
        bool motion_frame_valid = current_motion && 
                                  current_motion->timesteps > 0 && 
                                  current_frame >= 0 && 
                                  current_frame < static_cast<int>(current_motion->timesteps);

        // Independent capability checks for the current motion
        bool has_joint_data = motion_frame_valid && current_motion->GetNumJoints() >= 29;
        bool has_body_positions = motion_frame_valid && current_motion->GetNumBodies() >= 1;
        bool has_body_quaternions = motion_frame_valid && current_motion->GetNumBodyQuaternions() >= 1;

        // ---- Initialise target arrays with safe defaults ----
        std::array<double, 29> body_q_target;
        body_q_target.fill(0.0);
        std::array<double, 3> base_trans_target = {0.0, 0.0, 0.0};
        std::array<double, 4> base_quat_target = {1.0, 0.0, 0.0, 0.0};  // Identity quaternion
        
        // ---- Populate measured values from robot state (always available) ----
        // Remap from IsaacLab joint ordering to MuJoCo ordering and add default offsets.
        std::array<double, 29> body_q_measured;
        for (int i = 0; i < 29; i++) {
          body_q_measured[i] = state.body_q[isaaclab_to_mujoco[i]] + default_angles[i];
        }
        std::array<double, 3> base_trans_measured = {0.0, -1.0, 0.793};  // Fixed default position
        std::array<double, 4> base_quat_measured = state.base_quat;       // From IMU

        // Populate joint targets if available
        if (has_joint_data) {
          for (int i = 0; i < 29; i++) {
            body_q_target[i] = current_motion->JointPositions(current_frame)[isaaclab_to_mujoco[i]];
          }
        }

        // Populate body pose targets if available
        if (has_body_positions && has_body_quaternions) {
          base_trans_target = current_motion->BodyPositions(current_frame)[0];

          // Get heading state atomically (both quaternion and delta heading together)
          auto heading_state_data = heading_state_buffer.GetDataWithTime().data;
          HeadingState heading_state = heading_state_data ? *heading_state_data : HeadingState();
          
          // Calculate initial heading from robot root pose (using double precision functions)
          auto init_heading = calc_heading_quat_d(heading_state.init_base_quat);
          
          // Calculate inverse heading from reference data
          auto data_heading_inv = calc_heading_quat_inv_d(init_ref_data_root_rot_array);
          
          // Apply delta heading calculation
          auto apply_delta_heading = quat_mul_d(init_heading, data_heading_inv);
          
          // Apply additional delta heading if specified
          auto delta_quat = euler_z_to_quat_d(heading_state.delta_heading);
          apply_delta_heading = quat_mul_d(delta_quat, apply_delta_heading);
            
          // Get reference data root rotation at this target frame (first body part, index 0)
          const auto motion_body_quat = current_motion->BodyQuaternions(current_frame);
          std::array<double, 4> ref_data_root_rot_array = motion_body_quat[0];
          
          // Calculate new reference root rotation with heading applied
          base_quat_target = quat_mul_d(apply_delta_heading, ref_data_root_rot_array);

          // rotate the translation by the same quat:
          base_trans_target = quat_rotate_d(apply_delta_heading, base_trans_target);
        } else if (has_body_quaternions) {
          // Only have quaternions, use them without position
          auto heading_state_data = heading_state_buffer.GetDataWithTime().data;
          HeadingState heading_state = heading_state_data ? *heading_state_data : HeadingState();
          
          auto init_heading = calc_heading_quat_d(heading_state.init_base_quat);
          auto data_heading_inv = calc_heading_quat_inv_d(init_ref_data_root_rot_array);
          auto apply_delta_heading = quat_mul_d(init_heading, data_heading_inv);
          auto delta_quat = euler_z_to_quat_d(heading_state.delta_heading);
          apply_delta_heading = quat_mul_d(delta_quat, apply_delta_heading);
            
          const auto motion_body_quat = current_motion->BodyQuaternions(current_frame);
          std::array<double, 4> ref_data_root_rot_array = motion_body_quat[0];
          base_quat_target = quat_mul_d(apply_delta_heading, ref_data_root_rot_array);
        }
        // else: keep default values (zeros for position, identity for quaternion)

        // Rotate the vr_3point_position by the reference data root quat (or identity if motion invalid):
        auto vr_3point_position_sent = vr_3point_position;
        for (int i = 0; i < 3; i++) {
          std::array<double, 3> point_pos = {
            vr_3point_position_sent[i * 3 + 0],
            vr_3point_position_sent[i * 3 + 1], 
            vr_3point_position_sent[i * 3 + 2]
          };
          auto rotated_point = quat_rotate_d(base_quat_target, point_pos);
          vr_3point_position_sent[i * 3 + 0] = rotated_point[0];
          vr_3point_position_sent[i * 3 + 1] = rotated_point[1];
          vr_3point_position_sent[i * 3 + 2] = rotated_point[2];
        }

        // Write target pose:
        output_data_map_[kBaseTransTarget].assign(base_trans_target.begin(), base_trans_target.end());
        output_data_map_[kBaseQuatTarget].assign(base_quat_target.begin(), base_quat_target.end());
        output_data_map_[kBodyQTarget].assign(body_q_target.begin(), body_q_target.end());

        output_data_map_[kBaseTransMeasured].assign(base_trans_measured.begin(), base_trans_measured.end());
        output_data_map_[kBaseQuatMeasured].assign(base_quat_measured.begin(), base_quat_measured.end());
        output_data_map_[kBodyQMeasured].assign(body_q_measured.begin(), body_q_measured.end());
        output_data_map_[kLeftHandQMeasured].assign(left_hand_joint.begin(), left_hand_joint.end());
        output_data_map_[kRightHandQMeasured].assign(right_hand_joint.begin(), right_hand_joint.end());

        // write vr controller data:
        output_data_map_[kVr3pointPosition].assign(vr_3point_position_sent.begin(), vr_3point_position_sent.end());
        output_data_map_[kVr3pointOrientation].assign(vr_3point_orientation.begin(), vr_3point_orientation.end());
        output_data_map_[kVr3pointCompliance].assign(vr_3point_compliance.begin(), vr_3point_compliance.end());

        output_data_sbuf_.clear();
        msgpack::pack(output_data_sbuf_, output_data_map_);
    }

    /// Protected constructor – sub-classes must provide a StateLogger reference.
    explicit OutputInterface(StateLogger& logger) : state_logger_(logger) {}

    // Non-copyable, non-movable (instances are managed via unique_ptr).
    OutputInterface(const OutputInterface&) = delete;
    OutputInterface& operator=(const OutputInterface&) = delete;
    OutputInterface(OutputInterface&&) = delete;
    OutputInterface& operator=(OutputInterface&&) = delete;

    OutputType type_ = OutputType::UNKNOWN;  ///< Concrete output transport tag.

    /// Reference to the shared StateLogger (provides measured robot state).
    StateLogger& state_logger_;

    /// Reusable map populated by pack_output_data_map() each tick.
    std::map<std::string, std::vector<double>> output_data_map_;
    /// Reusable msgpack serialisation buffer (cleared and repacked each tick).
    msgpack::sbuffer output_data_sbuf_;

};

#endif // OUTPUT_INTERFACE_HPP

