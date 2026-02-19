/**
 * @file streamed_motion_merger.hpp
 * @brief Reusable sliding-window merger for streamed motion data.
 *
 * StreamedMotionMerger receives chunks of motion frames (joint positions /
 * velocities, body quaternions, SMPL data) from any streaming source (ZMQ,
 * ROS2, etc.) and merges them into a single growing MotionSequence using a
 * sliding-window approach.
 *
 * ## Key Concepts
 *
 * - **Frame indices**: Each incoming chunk carries a vector of monotonically
 *   increasing integer indices that identify frames in a global timeline.
 *   The merger uses these to align new data with the existing window.
 *
 * - **Frame step**: The stride between consecutive frame indices (e.g. 2 if
 *   the sender runs at 60 Hz and the consumer at 30 Hz).  Detected
 *   automatically from the first two indices of each chunk.
 *
 * - **Sliding window**: The merger maintains a window of frames centred
 *   around the current playback position.  It keeps `HISTORY_FRAMES` past
 *   frames for smooth interpolation and appends new frames from the
 *   incoming chunk.  Old frames that fall behind the window are discarded.
 *
 * - **Catch-up reset**: If the gap between the playback position and the
 *   incoming data exceeds `MAX_GAP_FRAMES`, the merger resets the window
 *   to the start of the incoming chunk and signals the caller to reset
 *   the playback cursor to frame 0.  This prevents unbounded buffering
 *   when the network falls behind.
 *
 * - **Protocol versions**: The merger itself is version-agnostic – it
 *   merges whatever data fields are present in IncomingData.  Protocol-
 *   version validation (rejecting changes mid-session, etc.) is left to
 *   the caller (e.g. ZMQEndpointInterface).
 *
 * ## Thread Safety
 *
 * The merger is **not** thread-safe.  All calls must be serialised by the
 * caller (typically by holding the data_mutex_ in ZMQEndpointInterface).
 */

#ifndef STREAMED_MOTION_MERGER_HPP
#define STREAMED_MOTION_MERGER_HPP

#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <iostream>
#include <iomanip>
#include <cstring>

// Forward declaration – MotionSequence is defined in motion_data_reader.hpp.
struct MotionSequence;

/**
 * @class StreamedMotionMerger
 * @brief Merges incoming motion-frame chunks into a sliding-window
 *        MotionSequence for real-time playback.
 */
class StreamedMotionMerger {
public:
    /// Compile-time toggle for debug log output.
    static constexpr bool DEBUG_LOGGING = true;
    /// Number of already-consumed frames to retain before the playback cursor
    /// (provides look-back for interpolation / blending).
    static constexpr int HISTORY_FRAMES = 5;
    /// Maximum tolerated gap (in current-rate frames) before a catch-up reset.
    static constexpr int MAX_GAP_FRAMES = 200;
    
    /// Returned by MergeIncomingData() to communicate what happened.
    struct MergeResult {
        std::shared_ptr<MotionSequence> motion;  ///< Merged motion (nullptr on failure).
        int window_start = 0;                    ///< Global frame index of motion[0].
        int frame_offset_adjustment = 0;         ///< Subtract from current_frame to compensate for window shift.
        bool did_catchup_reset = false;           ///< True → caller should reset playback to frame 0.
        int frame_step = 1;                       ///< Detected stride between consecutive frame indices.
        int protocol_version = 0;                 ///< Protocol version of the incoming data (1, 2, or 3).
    };
    
    /// All the data needed for one merge operation, decoded by the caller.
    struct IncomingData {
        // -- Joint data (required in v1 & v3, optional in v2) --
        std::vector<std::vector<double>> joint_pos;  ///< [frame][joint] positions (radians).
        std::vector<std::vector<double>> joint_vel;  ///< [frame][joint] velocities (rad/s).
        
        // -- Body quaternions (required for all versions) --
        std::vector<std::vector<std::array<double, 4>>> body_quat;  ///< [frame][body][w,x,y,z].
        
        // -- SMPL data (required in v2 & v3, optional in v1) --
        std::vector<std::vector<std::array<double, 3>>> smpl_joints;  ///< [frame][joint][x,y,z].
        std::vector<std::vector<std::array<double, 3>>> smpl_pose;    ///< [frame][pose][axis-angle x,y,z].
        
        std::vector<int64_t> frame_indices;  ///< Monotonic global frame indices (required).
        
        int protocol_version = 1;    ///< Protocol version (1, 2, or 3).
        bool catch_up_enabled = true; ///< true → use MAX_GAP_FRAMES; false → allow infinite delay.
        
        // Derived dimensions (must match the vector sizes above)
        int num_frames = 0;       ///< Number of frames in this chunk.
        int num_joints = 0;       ///< Joints per frame (joint_pos / joint_vel width).
        int num_quat_bodies = 0;  ///< Number of rigid bodies per frame (body_quat width).
        int num_smpl_joints = 0;  ///< SMPL joints per frame.
        int num_smpl_poses = 0;   ///< SMPL pose parameters per frame.
    };
    
    StreamedMotionMerger() {
        Reset();
    }
    
    // Reset the merger state (clear all buffered data)
    void Reset() {
        streamed_motion_ = std::make_shared<MotionSequence>();
        streamed_motion_->name = "streamed";
        streamed_motion_->ReserveCapacity(15000, 29, 1, 1, 0, 0);
        stream_window_start_ = 0;
    }
    
    // Main merging method: merge incoming data with existing buffered data
    // Returns MergeResult containing the merged motion and playback adjustments
    // 
    // Note: Protocol version validation should be done by the caller before calling this method.
    // The merger doesn't care about protocol versions - it just merges the data.
    MergeResult MergeIncomingData(const IncomingData& data, int current_playback_frame) {
        MergeResult result;
        
        // Validate incoming data
        if (!ValidateIncomingData(data)) {
            std::cerr << "[StreamedMotionMerger] Invalid incoming data" << std::endl;
            return result;
        }
        
        // Extract frame step and validate
        int frame_step = CalculateFrameStep(data.frame_indices);
        int incoming_frame_start = static_cast<int>(data.frame_indices[0]);
        int incoming_frame_end = static_cast<int>(data.frame_indices[data.num_frames - 1]);
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[StreamedMotionMerger] Processing " << data.num_frames << " frames, "
                      << "incoming_frame_start=" << incoming_frame_start 
                      << ", frame_step=" << frame_step << std::endl;
        }
        
        // Calculate sliding window parameters
        int global_playback_frame = stream_window_start_ + frame_step * std::max(0, current_playback_frame - HISTORY_FRAMES);
        int new_window_start = stream_window_start_;
        int merge_dst_frame = 0;
        bool did_catchup = false;
        
        CalculateSlidingWindow(
            incoming_frame_start,
            incoming_frame_end,
            frame_step,
            current_playback_frame,
            global_playback_frame,
            data.catch_up_enabled,
            new_window_start,
            merge_dst_frame,
            did_catchup
        );
        
        // Create new motion sequence
        auto new_motion = CreateNewMotion(data);
        
        // Copy old data to fill gap before incoming data
        if (merge_dst_frame > 0) {
            CopyOldDataToNewMotion(
                streamed_motion_,
                stream_window_start_,
                new_motion,
                new_window_start,
                incoming_frame_start,
                frame_step,
                data
            );
        }
        
        // Copy incoming data to new motion
        CopyIncomingDataToMotion(data, new_motion, merge_dst_frame);
        
        // Update total timesteps
        new_motion->timesteps = merge_dst_frame + data.num_frames;
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[StreamedMotionMerger] Merged motion: " << new_motion->timesteps 
                      << " frames (copied: " << merge_dst_frame << " + incoming: " << data.num_frames << ")" << std::endl;
        }
        
        // Calculate frame offset adjustment BEFORE updating state
        int old_window_start = stream_window_start_;
        int window_shift_ticks = new_window_start - old_window_start;
        int window_shift = (frame_step > 0) ? (window_shift_ticks / frame_step) : 0;
        
        // Update state
        streamed_motion_ = new_motion;
        stream_window_start_ = new_window_start;
        
        // Build result
        result.motion = new_motion;
        result.window_start = new_window_start;
        result.frame_offset_adjustment = did_catchup ? 0 : window_shift;
        result.did_catchup_reset = did_catchup;
        result.frame_step = frame_step;
        result.protocol_version = data.protocol_version;
        
        return result;
    }
    
private:
    std::shared_ptr<MotionSequence> streamed_motion_;
    int stream_window_start_ = 0;
    
    // Validate incoming data structure
    bool ValidateIncomingData(const IncomingData& data) const {
        // Check required fields
        if (data.body_quat.empty() || data.frame_indices.empty()) {
            std::cerr << "[StreamedMotionMerger] Missing required fields (body_quat or frame_indices)" << std::endl;
            return false;
        }
        
        // Validate protocol-specific requirements
        if (data.protocol_version == 3) {
            // Version 3: requires both SMPL data AND joint data
            if (data.smpl_joints.empty() || data.smpl_pose.empty()) {
                std::cerr << "[StreamedMotionMerger] Protocol v3 missing smpl_joints or smpl_pose" << std::endl;
                return false;
            }
            if (data.joint_pos.empty() || data.joint_vel.empty()) {
                std::cerr << "[StreamedMotionMerger] Protocol v3 missing joint_pos or joint_vel" << std::endl;
                return false;
            }
        } else if (data.protocol_version == 2) {
            // Version 2: requires SMPL data (joint data optional)
            if (data.smpl_joints.empty() || data.smpl_pose.empty()) {
                std::cerr << "[StreamedMotionMerger] Protocol v2 missing smpl_joints or smpl_pose" << std::endl;
                return false;
            }
        } else if (data.protocol_version == 1) {
            // Version 1: requires joint data (SMPL data optional)
            if (data.joint_pos.empty() || data.joint_vel.empty()) {
                std::cerr << "[StreamedMotionMerger] Protocol v1 missing joint_pos or joint_vel" << std::endl;
                return false;
            }
        } else {
            std::cerr << "[StreamedMotionMerger] Unsupported protocol version: " << data.protocol_version << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Calculate frame step from frame indices
    int CalculateFrameStep(const std::vector<int64_t>& frame_indices) const {
        if (frame_indices.size() < 2) {
            return 1;
        }
        int64_t step = std::abs(frame_indices[1] - frame_indices[0]);
        return step > 0 ? static_cast<int>(step) : 1;
    }
    
    // Calculate sliding window parameters
    void CalculateSlidingWindow(
        int incoming_frame_start,
        int incoming_frame_end,
        int frame_step,
        int current_playback_frame,
        int global_playback_frame,
        bool catch_up_enabled,
        int& new_window_start,
        int& merge_dst_frame,
        bool& did_catchup
    ) {
        // Special case: first packet
        if (!streamed_motion_ || streamed_motion_->timesteps <= 0) {
            new_window_start = incoming_frame_start;
            merge_dst_frame = 0;
            did_catchup = true;
            return;
        }
        
        // Calculate max gap based on catch_up flag
        int max_gap_frames = catch_up_enabled 
            ? (MAX_GAP_FRAMES + HISTORY_FRAMES) 
            : std::numeric_limits<int>::max();
        

        int stream_window_end = stream_window_start_ + frame_step * (streamed_motion_->timesteps - 1);

        if (DEBUG_LOGGING) {
            std::cout << "[StreamedMotionMerger] incoming_frame_start: " << incoming_frame_start
                      << ", incoming_frame_end: " << incoming_frame_end
                      << ", stream_window_start_: " << stream_window_start_
                      << ", stream_window_end: " << stream_window_end
                      << ", frame_step: " << frame_step
                      << ", global_playback_frame: " << global_playback_frame
                      << ", streamed_motion_->timesteps: " << streamed_motion_->timesteps
                      << std::endl;
        }

        // Check for incoming data older than current window
        if (incoming_frame_start <= stream_window_start_) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[StreamedMotionMerger] WARNING: incoming_frame_start (" << incoming_frame_start
                          << ") < stream_window_start_ (" << stream_window_start_ << ") - forcing catch-up" << std::endl;
            }
            new_window_start = incoming_frame_start;
            merge_dst_frame = 0;
            did_catchup = true;
            return;
        } else if (incoming_frame_end <= stream_window_end) {
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[StreamedMotionMerger] WARNING: incoming_frame_end (" << incoming_frame_end
                          << ") <= stream_window_end (" << stream_window_end << ") - forcing catch-up" << std::endl;
            }
            new_window_start = incoming_frame_start;
            merge_dst_frame = 0;
            did_catchup = true;
            return;
        }
        
        // Tentative window aligned to playback
        int desired_window_start = global_playback_frame;
        int tentative_window_start = std::min(desired_window_start, incoming_frame_start);
        int delta_to_incoming = incoming_frame_start - tentative_window_start;
        int tentative_merge_dst = (frame_step > 0) ? (delta_to_incoming / frame_step) : 0;
        
        // Check for large gap
        bool large_gap_from_old = incoming_frame_start > stream_window_end + frame_step;
        
        if (tentative_merge_dst > max_gap_frames || large_gap_from_old) {
            // Catch-up: reset window to incoming frame
            new_window_start = incoming_frame_start;
            merge_dst_frame = 0;
            did_catchup = true;
            
            if constexpr (DEBUG_LOGGING) {
                std::cout << "[StreamedMotionMerger] CATCH-UP: gap too large or old data expired" << std::endl;
            }
        } else {
            // Normal merge
            new_window_start = tentative_window_start;
            merge_dst_frame = tentative_merge_dst;
        }
    }
    
    // Create new motion sequence with appropriate capacity
    std::shared_ptr<MotionSequence> CreateNewMotion(const IncomingData& data) const {
        auto new_motion = std::make_shared<MotionSequence>();
        new_motion->name = "streamed";
        
        int joints_to_reserve = data.num_joints;
        int bodies_to_reserve = 1;
        int body_quaternions_to_reserve = data.num_quat_bodies;
        int smpl_joints_to_reserve = data.num_smpl_joints;
        int smpl_poses_to_reserve = data.num_smpl_poses;
        
        new_motion->ReserveCapacity(
            15000,
            joints_to_reserve,
            bodies_to_reserve,
            body_quaternions_to_reserve,
            smpl_joints_to_reserve,
            smpl_poses_to_reserve
        );
        
        // Initialize body_part_indexes (typically just root for streaming)
        new_motion->SetBodyPartIndexes({0});
        
        return new_motion;
    }
    
    // Copy old data to new motion to fill gap before incoming data
    void CopyOldDataToNewMotion(
        std::shared_ptr<MotionSequence> old_motion,
        int old_window_start,
        std::shared_ptr<MotionSequence> new_motion,
        int new_window_start,
        int incoming_frame_start,
        int frame_step,
        const IncomingData& data
    ) {
        if (!old_motion || old_motion->timesteps <= 0) {
            return;
        }
        
        int old_window_end = old_window_start + frame_step * old_motion->timesteps;
        
        // Find overlap between old data and needed range
        int need_start_global = new_window_start;
        int need_end_global = incoming_frame_start;
        int overlap_start_global = std::max(need_start_global, old_window_start);
        int overlap_end_global = std::min(need_end_global, old_window_end);
        
        if (overlap_start_global >= overlap_end_global) {
            return;  // No overlap
        }
        
        // Calculate copy parameters
        int start_offset_old = overlap_start_global - old_window_start;
        int start_offset_new = overlap_start_global - new_window_start;
        int overlap_span = overlap_end_global - overlap_start_global;
        int copy_src_idx = (frame_step > 0) ? (start_offset_old / frame_step) : 0;
        int copy_dst_idx = (frame_step > 0) ? (start_offset_new / frame_step) : 0;
        int copy_count = (frame_step > 0) ? (overlap_span / frame_step) : 0;
        
        if constexpr (DEBUG_LOGGING) {
            std::cout << "[StreamedMotionMerger] Copying old data: "
                      << "global [" << overlap_start_global << ".." << (overlap_end_global-1) << "] → "
                      << "new_motion[" << copy_dst_idx << ".." << (copy_dst_idx + copy_count - 1) << "]" << std::endl;
        }
        
        // Copy joint data if present
        if (data.num_joints > 0 && old_motion->GetNumJoints() > 0) {
            int joints_to_copy = std::min(data.num_joints, old_motion->GetNumJoints());
            for (int i = 0; i < copy_count; ++i) {
                for (int joint = 0; joint < joints_to_copy; ++joint) {
                    new_motion->JointPositions(copy_dst_idx + i)[joint] = 
                        old_motion->JointPositions(copy_src_idx + i)[joint];
                    new_motion->JointVelocities(copy_dst_idx + i)[joint] = 
                        old_motion->JointVelocities(copy_src_idx + i)[joint];
                }
            }
        }
        
        // Copy body quaternions
        int old_quat_bodies = old_motion->GetNumBodyQuaternions();
        int quat_bodies_to_copy = std::min(data.num_quat_bodies, old_quat_bodies);
        for (int i = 0; i < copy_count; ++i) {
            for (int b = 0; b < quat_bodies_to_copy; ++b) {
                for (int q = 0; q < 4; ++q) {
                    new_motion->BodyQuaternions(copy_dst_idx + i)[b][q] = 
                        old_motion->BodyQuaternions(copy_src_idx + i)[b][q];
                }
            }
        }
        
        // Copy SMPL data if present
        if (data.num_smpl_joints > 0 && old_motion->GetNumSmplJoints() > 0) {
            int smpl_joints_to_copy = std::min(data.num_smpl_joints, old_motion->GetNumSmplJoints());
            for (int i = 0; i < copy_count; ++i) {
                for (int joint = 0; joint < smpl_joints_to_copy; ++joint) {
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        new_motion->SmplJoints(copy_dst_idx + i)[joint][xyz] = 
                            old_motion->SmplJoints(copy_src_idx + i)[joint][xyz];
                    }
                }
            }
        }
        
        if (data.num_smpl_poses > 0 && old_motion->GetNumSmplPoses() > 0) {
            int smpl_poses_to_copy = std::min(data.num_smpl_poses, old_motion->GetNumSmplPoses());
            for (int i = 0; i < copy_count; ++i) {
                for (int p = 0; p < smpl_poses_to_copy; ++p) {
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        new_motion->SmplPoses(copy_dst_idx + i)[p][xyz] = 
                            old_motion->SmplPoses(copy_src_idx + i)[p][xyz];
                    }
                }
            }
        }
    }
    
    // Copy incoming data to motion sequence
    void CopyIncomingDataToMotion(
        const IncomingData& data,
        std::shared_ptr<MotionSequence> motion,
        int dst_frame_offset
    ) {
        // Copy joint data if present
        if (!data.joint_pos.empty() && !data.joint_vel.empty()) {
            for (int frame = 0; frame < data.num_frames; ++frame) {
                for (int joint = 0; joint < data.num_joints; ++joint) {
                    motion->JointPositions(dst_frame_offset + frame)[joint] = data.joint_pos[frame][joint];
                    motion->JointVelocities(dst_frame_offset + frame)[joint] = data.joint_vel[frame][joint];
                }
            }
        }
        
        // Copy body quaternions (always present)
        for (int frame = 0; frame < data.num_frames; ++frame) {
            for (int body = 0; body < data.num_quat_bodies; ++body) {
                for (int q = 0; q < 4; ++q) {
                    motion->BodyQuaternions(dst_frame_offset + frame)[body][q] = 
                        data.body_quat[frame][body][q];
                }
            }
        }
        
        // Copy SMPL joints if present
        if (!data.smpl_joints.empty()) {
            for (int frame = 0; frame < data.num_frames; ++frame) {
                for (int joint = 0; joint < data.num_smpl_joints; ++joint) {
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        motion->SmplJoints(dst_frame_offset + frame)[joint][xyz] = 
                            data.smpl_joints[frame][joint][xyz];
                    }
                }
            }
        }
        
        // Copy SMPL poses if present
        if (!data.smpl_pose.empty()) {
            for (int frame = 0; frame < data.num_frames; ++frame) {
                for (int pose = 0; pose < data.num_smpl_poses; ++pose) {
                    for (int xyz = 0; xyz < 3; ++xyz) {
                        motion->SmplPoses(dst_frame_offset + frame)[pose][xyz] = 
                            data.smpl_pose[frame][pose][xyz];
                    }
                }
            }
        }
    }
};

#endif // STREAMED_MOTION_MERGER_HPP

