/**
 * @file localmotion_kplanner.hpp
 * @brief Locomotion planner base class, locomotion modes, and movement state.
 *
 * This header defines:
 *  - **LocomotionMode** enum – all supported locomotion styles (idle, walk,
 *    run, squat, kneel, crawl, boxing, styled walks, etc.) and helper
 *    predicates (is_static, is_standing, is_squat) and motion-set selectors.
 *  - **PlannerState** – enable / initialised flags.
 *  - **MovementState** – per-tick locomotion command (mode, direction, facing,
 *    speed, height) passed via DataBuffer from input interfaces to the
 *    planner thread.
 *  - **PlannerConfig** – model path, version, look-ahead, default height, seed.
 *  - **PlannerTiming** – timing breakdown for profiling planner phases.
 *  - **LocalMotionPlannerBase** – abstract base class providing:
 *    - `Initialize()` – set up context and run initial inference.
 *    - `UpdatePlanning()` – per-tick replanning with new movement commands.
 *    - `ResampleGeneratedSequence50Hz()` – convert 30 Hz planner output to
 *      50 Hz control-rate MotionSequence.
 *    - Pure-virtual hooks for backend-specific work: `InitializeSpecific()`,
 *      `RunInference()`, `UpdateInputTensors()`, and buffer accessors.
 *
 * Concrete backends: `LocalMotionPlannerONNX` (ONNX Runtime) and
 * `LocalMotionPlannerTensorRT` (TensorRT).
 */

#ifndef LOCALMOTION_KPLANNER_HPP
#define LOCALMOTION_KPLANNER_HPP

#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <mutex>

// Motion data structures
#include "motion_data_reader.hpp"
#include "robot_parameters.hpp"
#include "policy_parameters.hpp"
#include "math_utils.hpp"
#include "utils.hpp"
#include "cnpy.h"

/// Planner lifecycle flags (set by input interfaces, read by planner thread).
struct PlannerState {
    bool enabled = false;       ///< True when the planner should be running.
    bool initialized = false;   ///< True after the first successful inference.
};

/**
 * @brief Per-tick locomotion command passed from input interfaces to the planner.
 *
 * Written into a DataBuffer<MovementState> by the input thread and read
 * by the planner thread each planning cycle.
 */
struct MovementState {
    int locomotion_mode;                         ///< LocomotionMode cast to int.
    std::array<double, 3> movement_direction;    ///< Unit vector: desired movement direction [x,y,z].
    std::array<double, 3> facing_direction;      ///< Unit vector: desired facing direction [x,y,z].
    double movement_speed;                        ///< Desired speed (−1 = mode default, 0 = stationary).
    double height;                                ///< Desired body height (−1 = mode default).

    MovementState(int mode = 0, 
                     const std::array<double, 3>& movement = {0.0, 0.0, 0.0},
                     const std::array<double, 3>& facing = {1.0, 0.0, 0.0},
                     double speed = 0.0,
                     double height = 0.0)
        : locomotion_mode(mode), movement_direction(movement), facing_direction(facing), movement_speed(speed), height(height) {}
  };

/**
 * @brief Enumeration for different locomotion modes used by the planner
 */
 enum class LocomotionMode {
    IDLE = 0,
    SLOW_WALK = 1, // 0.1m/s ~ 0.8m/s
    WALK = 2, // 0.8m/s ~ 2.5m/s
    RUN = 3, // 2.5m/s ~ 7.5m/s
    IDEL_SQUAT = 4,
    IDEL_KNEEL_TWO_LEGS = 5,
    IDEL_KNEEL = 6,
    IDEL_LYING_FACE_DOWN = 7,
    CRAWLING = 8,
    IDEL_BOXING = 9,
    WALK_BOXING = 10,
    LEFT_PUNCH = 11,
    RIGHT_PUNCH = 12,
    RANDOM_PUNCH = 13,
    ELBOW_CRAWLING = 14,
    LEFT_HOOK = 15,
    RIGHT_HOOK = 16,
    FORWARD_JUMP = 17,
    STEALTH_WALK = 18,
    INJURED_WALK = 19,
    LEDGE_WALKING = 20,
    OBJECT_CARRYING = 21,
    STEALTH_WALK_2 = 22,
    HAPPY_DANCE_WALK = 23,
    ZOMBIE_WALK = 24,
    GUN_WALK = 25,
    SCARE_WALK = 26,
};

inline constexpr bool is_static_motion_mode(LocomotionMode mode) {
    return mode == LocomotionMode::IDLE || 
           mode == LocomotionMode::IDEL_SQUAT || 
           mode == LocomotionMode::IDEL_KNEEL_TWO_LEGS || 
           mode == LocomotionMode::IDEL_KNEEL || 
           mode == LocomotionMode::IDEL_LYING_FACE_DOWN || 
           mode == LocomotionMode::IDEL_BOXING;
}

inline constexpr bool is_standing_motion_mode(LocomotionMode mode) {
    return mode == LocomotionMode::IDLE || 
           mode == LocomotionMode::SLOW_WALK || 
           mode == LocomotionMode::WALK || 
           mode == LocomotionMode::RUN ||
           mode == LocomotionMode::IDEL_BOXING ||
           mode == LocomotionMode::WALK_BOXING ||
           mode == LocomotionMode::LEFT_PUNCH ||
           mode == LocomotionMode::RIGHT_PUNCH ||
           mode == LocomotionMode::RANDOM_PUNCH ||
           mode == LocomotionMode::LEFT_HOOK ||
           mode == LocomotionMode::RIGHT_HOOK ||
           mode == LocomotionMode::FORWARD_JUMP ||
           mode == LocomotionMode::STEALTH_WALK ||
           mode == LocomotionMode::INJURED_WALK ||
           mode == LocomotionMode::LEDGE_WALKING ||
           mode == LocomotionMode::OBJECT_CARRYING ||
           mode == LocomotionMode::STEALTH_WALK_2 ||
           mode == LocomotionMode::HAPPY_DANCE_WALK ||
           mode == LocomotionMode::ZOMBIE_WALK ||
           mode == LocomotionMode::GUN_WALK ||
           mode == LocomotionMode::SCARE_WALK;
}

inline constexpr bool is_squat_motion_mode(LocomotionMode mode) {
    return mode == LocomotionMode::IDEL_SQUAT ||
           mode == LocomotionMode::IDEL_KNEEL_TWO_LEGS ||
           mode == LocomotionMode::IDEL_KNEEL ||
           mode == LocomotionMode::IDEL_LYING_FACE_DOWN || 
           mode == LocomotionMode::CRAWLING ||
           mode == LocomotionMode::ELBOW_CRAWLING;

}

inline std::vector<LocomotionMode> get_standing_motion_modes() {
    return {LocomotionMode::SLOW_WALK, 
        LocomotionMode::WALK, 
        LocomotionMode::RUN,
        LocomotionMode::FORWARD_JUMP,
        LocomotionMode::STEALTH_WALK,
        LocomotionMode::INJURED_WALK,
    };
}

inline std::vector<LocomotionMode> get_styled_walking_motion_modes() {
    return {LocomotionMode::LEDGE_WALKING,
        LocomotionMode::OBJECT_CARRYING,
        LocomotionMode::STEALTH_WALK_2,
        LocomotionMode::HAPPY_DANCE_WALK,
        LocomotionMode::ZOMBIE_WALK,
        LocomotionMode::GUN_WALK,
        LocomotionMode::SCARE_WALK,
    };
}

inline std::vector<LocomotionMode> get_squat_motion_modes() {
    return {LocomotionMode::IDEL_SQUAT, 
        LocomotionMode::IDEL_KNEEL_TWO_LEGS, 
        LocomotionMode::IDEL_KNEEL, 
        // LocomotionMode::IDEL_LYING_FACE_DOWN, // TODO: uncomment this when we have a better lying face down motion
        LocomotionMode::CRAWLING,
        LocomotionMode::ELBOW_CRAWLING,
    };
}

inline std::vector<LocomotionMode> get_boxing_motion_modes() {
    return {LocomotionMode::IDEL_BOXING, 
        LocomotionMode::WALK_BOXING,
        LocomotionMode::LEFT_PUNCH,
        LocomotionMode::RIGHT_PUNCH,
        LocomotionMode::RANDOM_PUNCH,
        LocomotionMode::LEFT_HOOK,
        LocomotionMode::RIGHT_HOOK,
    };
}

inline std::vector<LocomotionMode> get_motion_set(int motion_set_index) {
    switch(motion_set_index) {
        case 0:
            return get_standing_motion_modes();
        case 1:
            return get_squat_motion_modes();
        case 2:
            return get_boxing_motion_modes();
        case 3:
            return get_styled_walking_motion_modes();
        default:
            std::cout << "✗ Error: Invalid motion set index" << std::endl;
            std::cout << "Using standing motion set" << std::endl;
            return get_standing_motion_modes();
        }
}
/**
 * @brief Configuration for planner initialization
 */
struct PlannerConfig {
    std::string model_path = "";
    int version = 0;
    int motion_look_ahead_steps = 2;      // 50Hz: 0.02s * 2 = 0.04s
    double default_height = 0.788740;      // Default robot height
    int initial_random_seed = 1234;        // Initial random seed for planner
};

/**
 * @brief Structure containing timing information for different planner phases
 */
struct PlannerTiming {
    std::chrono::microseconds gather_input_duration{0};
    std::chrono::microseconds model_duration{0};
    std::chrono::microseconds extract_duration{0};
    std::chrono::microseconds total_duration{0};
};

/**
 * @brief Base class containing common functionality for motion planners
 * 
 * This class provides shared data members, utility functions, and common logic
 * that both ONNX and TensorRT planners can inherit from or use.
 */
class LocalMotionPlannerBase {
public:
    /**
     * @brief Constructor with common configuration
     */
    LocalMotionPlannerBase(const PlannerConfig& config = PlannerConfig())
        : config_(config),
          current_random_seed_(config.initial_random_seed)
    {
        planner_motion_50hz_.ReserveCapacity(1500, 29, 1, 1, 0, 0);
        gen_frame_ = 0;
        std::cout << "LocalMotionPlannerBase constructor" << std::endl;
        std::cout << "config_.version: " << config_.version << std::endl;
        std::cout << "config_.motion_look_ahead_steps: " << config_.motion_look_ahead_steps << std::endl;
        std::cout << "config_.default_height: " << config_.default_height << std::endl;
        std::cout << "config_.initial_random_seed: " << config_.initial_random_seed << std::endl;
        std::cout << "Available modes number: " << GetValidModeValueRange() << std::endl;
    }

    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~LocalMotionPlannerBase() = default;

    // Public state flags - can be accessed directly by derived classes
    PlannerState planner_state_;
    PlannerTiming last_timing_;

    // Thread safety for motion state buffer access (public for external access)
    mutable std::mutex planner_motion_mutex_;

    // Generated motion sequence and the frame at which it was generated
    MotionSequence planner_motion_50hz_;
    int gen_frame_;
    bool motion_available_ = false;

protected:
    // Common configuration
    PlannerConfig config_;

public:

    // Current input state
    int current_random_seed_;


    /**
     * @brief Common validation for UpdatePlanning inputs
     * @return true if inputs are valid for planning
     */
    bool ValidatePlanningInputs() const {
        return planner_state_.initialized && planner_state_.enabled;
    }

    int GetValidModeValueRange() const {
        if (config_.version == 2)
        {
            return 27; // IDLE, SLOW_WALK, WALK, RUN, IDEL_SQUAT, IDEL_KNEEL_TWO_LEGS, IDEL_KNEEL, IDEL_LYING, IDEL_CRAWLING, IDEL_BOXING, WALK_BOXING, LEFT_PUNCH, RIGHT_PUNCH, RANDOM_PUNCH, ELBOW_CRAWLING, LEFT_HOOK, RIGHT_HOOK, FORWARD_JUMP, STEALTH_WALK, INJURED_WALK, LEDGE_WALKING, OBJECT_CARRYING, STEALTH_WALK_2, HAPPY_DANCE_WALK, ZOMBIE_WALK, GUN_WALK, SCARE_WALK
        }
        else if (config_.version == 1)
        {
            return 20; // IDLE, SLOW_WALK, WALK, RUN, IDEL_SQUAT, IDEL_KNEEL_TWO_LEGS, IDEL_KNEEL, IDEL_LYING, IDEL_CRAWLING, IDEL_BOXING, WALK_BOXING, LEFT_PUNCH, RIGHT_PUNCH, RANDOM_PUNCH, ELBOW_CRAWLING, LEFT_HOOK, RIGHT_HOOK, FORWARD_JUMP, STEALTH_WALK, INJURED_WALK
        }
        else if (config_.version == 0)
        {
            return 4; // IDLE, SLOW_WALK, WALK, RUN
        }
        else
        {
            std::cout << "✗ Error: Unsupported model version" << std::endl;
            throw std::runtime_error("Unsupported model version: " + std::to_string(config_.version));
        }
    }

public:

    /**
     * @brief Initialize planner with robot state and populate initial trajectory
     * 
     * COORDINATE FRAME NORMALIZATION STRATEGY:
     * To ensure consistent neural network behavior regardless of robot's initial orientation,
     * all data is normalized to a coordinate frame where the first context frame faces zero yaw:
     * 
     * 1. Extract yaw from robot's current orientation
     * 2. Compute rotation to make first frame face zero yaw  
     * 3. Apply same rotation to: input tensors, context frames, and trajectory data
     * 4. Neural network processes normalized data
     * 5. Results are rotated back to world coordinates using inverse rotation
     * 
     * This maintains mathematical consistency while improving model generalization.
     * 
     * @param base_quat Robot base quaternion [w,x,y,z]
     * @param joint_positions Current joint positions
     * @param planner_motion_state Motion state buffer to populate with initial trajectory
     * @return true if initialization successful, false otherwise
     */
    bool Initialize(
        const std::array<double, 4>& base_quat,
        const std::array<double, 29>& joint_positions) {

        // Reset initialization flag only; preserve enabled state set by external controller
        planner_state_.initialized = false;

        if(!InitializeSpecific()) {
            return false;
        }

        // ==== BEGIN FRESH INITIALIZATION ====
        // Initialize input vectors with defaults aligned to robot's current orientation
        // Extract yaw from robot's base quaternion to set proper facing direction
        UpdateInputTensors(static_cast<int>(LocomotionMode::IDLE),
            -1.0,
            -1.0,
            {0.0f, 0.0f, 0.0f},  // No movement
            {1.0f, 0.0f, 0.0f},  // Face in robot's current yaw direction
            current_random_seed_
        );

        // Initialize context directly
        gen_frame_ = 0;
        InitializeContext(base_quat, joint_positions);

        // Run initial inference with default parameters to populate motion data
        try {
            // Start timing model inference
            auto model_start_time = std::chrono::high_resolution_clock::now();

            RunInference();

            auto model_end_time = std::chrono::high_resolution_clock::now();

            // Start timing data extraction
            auto extract_start_time = std::chrono::high_resolution_clock::now();

            ResampleGeneratedSequence50Hz();

            auto extract_end_time = std::chrono::high_resolution_clock::now();
            
            // Log initialization timing
            auto model_duration = std::chrono::duration_cast<std::chrono::microseconds>(model_end_time - model_start_time);
            auto extract_duration = std::chrono::duration_cast<std::chrono::microseconds>(extract_end_time - extract_start_time);

            std::cout << "Planner Init timing - Model: " << model_duration.count() << "us"
                    << ", Extract: " << extract_duration.count() << "us" << std::endl;

            // Mark as fully initialized only after all setup is complete
            planner_state_.initialized = true;

            std::cout << "Planner initialized" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "✗ Error during planner initialization: " << e.what() << std::endl;
            planner_state_.initialized = false;
            return false;
        }

        return true;
    }

    bool ResampleGeneratedSequence50Hz()
    {
        int32_t timesteps_30hz = GetNumPredFrames();
        auto mujoco_qpos_data = GetMujocoQposBuffer();

        if(timesteps_30hz > 64)
        {
            std::cout << "✗ Error: Too many timesteps generated by planner" << std::endl;

            /*
            for(int i = 0; i < (G1_NUM_MOTOR + 7) * timesteps_30hz; i++)
            {
                std::cout << mujoco_qpos_data[i] << " " << std::isnan(mujoco_qpos_data[i]) << ",";
            }
            std::cout << std::endl;

            cnpy::npy_save("mode.npy", std::vector<int64_t>{GetModeValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("target_vel.npy", std::vector<float>{GetTargetVelValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("movement_direction.npy", GetMovementDirectionValues(), std::vector<size_t>{3}, "w");
            cnpy::npy_save("facing_direction.npy", GetFacingDirectionValues(), std::vector<size_t>{3}, "w");
            cnpy::npy_save("random_seed.npy", std::vector<int64_t>{GetRandomSeedValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("context_qpos.npy", GetContextBuffer(), std::vector<size_t>{4, (G1_NUM_MOTOR + 7)}, "w");
            */

            return false;
        }
        
        if(
            std::any_of(
                mujoco_qpos_data, mujoco_qpos_data + (G1_NUM_MOTOR + 7) * timesteps_30hz,
                [](float x) { return std::isnan(x); }
            )
        )
        {
            std::cout << "✗ Error: Mujoco qpos data contains nans" << std::endl;

            /*
            for(int i = 0; i < (G1_NUM_MOTOR + 7) * timesteps_30hz; i++)
            {
                std::cout << mujoco_qpos_data[i] << " " << std::isnan(mujoco_qpos_data[i]) << ",";
            }
            std::cout << std::endl;

            cnpy::npy_save("mode.npy", std::vector<int64_t>{GetModeValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("target_vel.npy", std::vector<float>{GetTargetVelValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("movement_direction.npy", GetMovementDirectionValues(), std::vector<size_t>{3}, "w");
            cnpy::npy_save("facing_direction.npy", GetFacingDirectionValues(), std::vector<size_t>{3}, "w");
            cnpy::npy_save("random_seed.npy", std::vector<int64_t>{GetRandomSeedValue()}.data(), std::vector<size_t>{1}, "w");
            cnpy::npy_save("context_qpos.npy", GetContextBuffer(), std::vector<size_t>{4, (G1_NUM_MOTOR + 7)}, "w");
            */

            return false;
        }

        // How long is the generated sequence in seconds?
        double motion_seconds = double(timesteps_30hz) / 30.0;

        // How many 50Hz timesteps are there?
        std::lock_guard<std::mutex> lock(planner_motion_mutex_);
        planner_motion_50hz_.timesteps = static_cast<int32_t>(std::floor(motion_seconds * 50));

        for(int32_t f_50hz = 0; f_50hz < planner_motion_50hz_.timesteps; ++f_50hz)
        {
            // what time are we sampling at?
            double t = double(f_50hz) / 50.0;

            // what (non integer) 30fps frame is this?
            double f_30hz = t * 30;

            int32_t f0 = static_cast<int32_t>(std::floor(f_30hz));
            int32_t f1 = std::min(f0 + 1, timesteps_30hz - 1);

            double w0 = 1.0 - (f_30hz - f0);
            double w1 = 1.0 - w0;

            // resample global body position:
            planner_motion_50hz_.BodyPositions(f_50hz)[0] = std::array<double, 3> {
                w0 * mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 0] + w1 * mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 0],
                w0 * mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 1] + w1 * mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 1],
                w0 * mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 2] + w1 * mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 2]
            };
            
            // resample global body quaternion:
            std::array<double, 4> q0 = {
                mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 3],
                mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 4],
                mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 5],
                mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 6]
            };
            std::array<double, 4> q1 = {
                mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 3],
                mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 4],
                mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 5],
                mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 6]
            };
            planner_motion_50hz_.BodyQuaternions(f_50hz)[0] = quat_slerp_d(q0, q1, f_30hz - f0);

            // resample joint angles:
            for(int joint = 0; joint < 29; ++joint) {
                planner_motion_50hz_.JointPositions(f_50hz)[joint] =
                    w0 * mujoco_qpos_data[f0 * (G1_NUM_MOTOR + 7) + 7 + mujoco_to_isaaclab[joint]] + 
                    w1 * mujoco_qpos_data[f1 * (G1_NUM_MOTOR + 7) + 7 + mujoco_to_isaaclab[joint]];
            }
        }

        // Update joint velocities using optimized frame access (no redundant resize)
        for(int32_t frame = 0; frame < planner_motion_50hz_.timesteps-1; ++frame)
        {
            for(int joint = 0; joint < 29; ++joint)
            {
                planner_motion_50hz_.JointVelocities(frame)[joint] = (
                    planner_motion_50hz_.JointPositions(frame+1)[joint] -
                    planner_motion_50hz_.JointPositions(frame)[joint]
                ) * 50.0;
            }
        }

        // Handle last frame velocity
        for(int joint = 0; joint < 29; ++joint)
        {
            planner_motion_50hz_.JointVelocities(planner_motion_50hz_.timesteps-1)[joint] = planner_motion_50hz_.JointVelocities(planner_motion_50hz_.timesteps-2)[joint];
        }
        motion_available_ = true;
        return true;
    }



    /**
     * @brief Update planner with new planning request and update motion state directly
     * @param existing_motion_50hz Existing motion sequence
     * @param gen_frame Frame at which the new motion sequence was generated
     * @param mode_value Locomotion mode (0=IDLE, 1=SLOW_WALK, 2=WALK, 3=RUN)
     * @param movement_direction Movement direction vector [x, y, z]
     * @param facing_direction Facing direction vector [x, y, z]
     * @param random_seed Optional random seed (if -1, use default)
     * @return true if replanning successful and buffer updated, false otherwise
     */
     bool UpdatePlanning(
        int gen_frame,
        const std::shared_ptr<const MotionSequence>& motion_sequence,
        int mode_value,
        float target_vel,
        float target_height,
        const std::array<float, 3>& movement_direction,
        const std::array<float, 3>& facing_direction,
        int random_seed = -1)
    {

        auto total_start_time = std::chrono::steady_clock::now();

        if (!ValidatePlanningInputs()) {
            return false;
        }

        // ===== GATHER INPUT PHASE =====
        auto gather_input_start_time = std::chrono::steady_clock::now();

        // Update input tensors with new values
        UpdateInputTensors(mode_value, target_vel, target_height, movement_direction, facing_direction, random_seed);
        
        // Update context and get frame information
        gen_frame_ = gen_frame + config_.motion_look_ahead_steps;
        UpdateContextFromMotion(motion_sequence);

        auto gather_input_end_time = std::chrono::steady_clock::now();

        // ===== MODEL INFERENCE PHASE =====
        auto model_start_time = std::chrono::steady_clock::now();

        // Run inference
        RunInference();

        auto model_end_time = std::chrono::steady_clock::now();

        // ===== EXTRACT PHASE =====
        auto extract_start_time = std::chrono::steady_clock::now();

        if(!ResampleGeneratedSequence50Hz())
        {
            return false;
        }

        auto extract_end_time = std::chrono::steady_clock::now();

        // ===== RECORD TIMING =====
        auto total_end_time = std::chrono::steady_clock::now();

        last_timing_.gather_input_duration = std::chrono::duration_cast<std::chrono::microseconds>(gather_input_end_time - gather_input_start_time);
        last_timing_.model_duration = std::chrono::duration_cast<std::chrono::microseconds>(model_end_time - model_start_time);
        last_timing_.extract_duration = std::chrono::duration_cast<std::chrono::microseconds>(extract_end_time - extract_start_time);
        last_timing_.total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);

        return true;
    }

    void InitializeContext(
        const std::array<double, 4>& base_quat,
        const std::array<double, 29>& joint_positions
    )
    {
        auto context_qpos_values = GetContextBuffer();
        
        // ROTATION STRATEGY: Normalize coordinate frame so first context frame faces zero yaw
        // This ensures consistent neural network input regardless of robot's initial orientation
        
        // Create zero facing direction quaternion
        std::array<double, 4> quat = {1.0, 0.0, 0.0, 0.0};

        for(int n = 0; n < 4; ++n) {
            int index_base = n * (G1_NUM_MOTOR + 7);

            // Set global position (default standing position)
            context_qpos_values[index_base + 0] = 0.0f;  // x = 0
            context_qpos_values[index_base + 1] = 0.0f;  // y = 0  
            context_qpos_values[index_base + 2] = static_cast<float>(config_.default_height);  // z = standing height

            // Set quaternion - apply same rotation as input tensors for consistency
            // This ensures all 4 context frames are rotated to the same zero-yaw coordinate frame
            context_qpos_values[index_base + 3] = static_cast<float>(quat[0]); // w
            context_qpos_values[index_base + 4] = static_cast<float>(quat[1]); // x
            context_qpos_values[index_base + 5] = static_cast<float>(quat[2]); // y
            context_qpos_values[index_base + 6] = static_cast<float>(quat[3]); // z

            // Set joint positions
            for (int i = 0; i < G1_NUM_MOTOR; i++) {
                context_qpos_values[index_base + 7 + i] = static_cast<float>(joint_positions[i]);
            }
        }
    }


    // Base class virtual method implementations  
    void UpdateContextFromMotion(const std::shared_ptr<const MotionSequence>& motion_sequence) {

        std::lock_guard<std::mutex> lock(planner_motion_mutex_);

        // Guard against race condition - motion not yet populated by control thread
        if (!motion_sequence || motion_sequence->timesteps == 0) {
            throw std::runtime_error("UpdateContextFromMotion: motion not ready, cannot update context");
        }
        
        // what time are we sampling at?
        double gen_time = double(gen_frame_) / 50.0;

        // now we need to sample 4 frames at 30hz, starting at gen_time:
        auto context_qpos_values = GetContextBuffer();
        for(int n=0; n<4; ++n)
        {
            // what time shall we sample the current motion sequence at?
            double t = gen_time + double(n) / 30.0;

            double f_50hz = t * 50.0;
            int f0 = static_cast<int>(std::floor(f_50hz));
            f0 = std::min(f0, motion_sequence->timesteps - 1);
            int f1 = std::min(f0 + 1, motion_sequence->timesteps - 1);

            double w0 = 1.0 - (f_50hz - f0);
            double w1 = 1.0 - w0;
            
            // resample global quaternion:
            std::array<double, 4> q0 = {motion_sequence->BodyQuaternions(f0)[0][0], motion_sequence->BodyQuaternions(f0)[0][1], motion_sequence->BodyQuaternions(f0)[0][2], motion_sequence->BodyQuaternions(f0)[0][3]};
            std::array<double, 4> q1 = {motion_sequence->BodyQuaternions(f1)[0][0], motion_sequence->BodyQuaternions(f1)[0][1], motion_sequence->BodyQuaternions(f1)[0][2], motion_sequence->BodyQuaternions(f1)[0][3]};
            std::array<double, 4> quat = quat_slerp_d(q0, q1, f_50hz - f0);

            std::copy(quat.begin(), quat.end(), &context_qpos_values[n * (G1_NUM_MOTOR + 7) + 3]);

            // resample global position:
            std::array<double, 3> p = {
                w0 * motion_sequence->BodyPositions(f0)[0][0] + w1 * motion_sequence->BodyPositions(f1)[0][0],
                w0 * motion_sequence->BodyPositions(f0)[0][1] + w1 * motion_sequence->BodyPositions(f1)[0][1],
                w0 * motion_sequence->BodyPositions(f0)[0][2] + w1 * motion_sequence->BodyPositions(f1)[0][2]
            };
            
            context_qpos_values[n * (G1_NUM_MOTOR + 7) + 0] = p[0];
            context_qpos_values[n * (G1_NUM_MOTOR + 7) + 1] = p[1];
            context_qpos_values[n * (G1_NUM_MOTOR + 7) + 2] = p[2];

            // resample joint positions
            for(int i = 0; i < G1_NUM_MOTOR; i++) {
                context_qpos_values[n * (G1_NUM_MOTOR + 7) + 7 + mujoco_to_isaaclab[i]] = w0 * motion_sequence->JointPositions(f0)[i] + w1 * motion_sequence->JointPositions(f1)[i];
            }
        }
    }

    /**
     * @brief Common method signatures that derived classes should implement
     */

    virtual bool InitializeSpecific() = 0;

    virtual void RunInference() = 0;

    virtual void UpdateInputTensors(int mode_value,
        float target_vel,
        float target_height,
        const std::array<float, 3>& movement_direction,
        const std::array<float, 3>& facing_direction,
        int random_seed) = 0;


    virtual float *GetContextBuffer() = 0;
    virtual int32_t GetNumPredFrames() = 0;
    virtual const float *GetMujocoQposBuffer() = 0;
    virtual float *GetMovementDirectionValues() = 0;
    virtual float *GetFacingDirectionValues() = 0;
    virtual float GetTargetVelValue() = 0;
    virtual float GetHeightValue() = 0;
    virtual int32_t GetRandomSeedValue() = 0;
    virtual int32_t GetModeValue() = 0;


};

#endif // LOCALMOTION_KPLANNER_HPP
