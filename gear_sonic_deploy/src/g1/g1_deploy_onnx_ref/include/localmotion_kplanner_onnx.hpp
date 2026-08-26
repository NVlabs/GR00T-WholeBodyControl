/**
 * @file localmotion_kplanner_onnx.hpp
 * @deprecated This file is NOT used in the current build.  The active planner
 *             backend is LocalMotionPlannerTensorRT (localmotion_kplanner_tensorrt.hpp).
 *             This file is kept for reference only.
 *
 * @brief ONNX Runtime backend for the locomotion planner.
 *
 * LocalMotionPlannerONNX is a concrete implementation of LocalMotionPlannerBase
 * that runs planner inference on the CPU via ONNX Runtime.  It is the simpler
 * of the two backends (the other being TensorRT) and is useful for development
 * / debugging or when a GPU is not available.
 *
 * ## Model Versions
 *
 *   Version | Inputs | Notes
 *   --------|--------|------
 *   0       | 6      | Basic: context, mode, target_vel, movement/facing direction, random_seed.
 *   1       | 11     | Adds: height, has_specific_target, specific_target_positions/headings, allowed_pred_num_tokens.
 *
 * ## Data Flow
 *
 *   1. Constructor loads the ONNX model and validates input tensor names.
 *   2. `InitializeSpecific()` allocates input buffers and creates ONNX tensors
 *      that alias the buffer memory (zero-copy).
 *   3. `UpdateInputTensors()` writes new locomotion commands into the buffers.
 *   4. `RunInference()` executes `session.Run()` synchronously on the CPU.
 *   5. Output pointers (`mujoco_qpos_data_`, `num_pred_frames_data_`) are
 *      captured from the output tensors for the base class to resample at 50 Hz.
 */

#ifndef LOCALMOTION_KPLANNER_ONNX_HPP
#define LOCALMOTION_KPLANNER_ONNX_HPP

#include <onnxruntime_cxx_api.h>
#include <iostream>

#include "localmotion_kplanner.hpp"
#include "ort_session.hpp"

/**
 * @class LocalMotionPlannerONNX
 * @brief ONNX Runtime backend for the locomotion planner.
 *
 * Runs inference synchronously on the CPU.  Input tensors alias the member
 * buffers so `UpdateInputTensors()` is effectively zero-copy.
 */
class LocalMotionPlannerONNX : public LocalMotionPlannerBase {
public:
    /**
     * @brief Constructor for LocalMotionPlannerONNX
     * @param env ONNX runtime environment
     * @param allocator ONNX allocator for memory management
     * @param config Planner configuration parameters
     */
    LocalMotionPlannerONNX(Ort::Env& env, 
                           Ort::AllocatorWithDefaultOptions& allocator,
                           const PlannerConfig& config = PlannerConfig())
        : LocalMotionPlannerBase(config), env_(env), allocator_(allocator) {
        
        if (!config_.model_path.empty()) {
            planner_session_ = std::make_shared<OrtSession>(config_.model_path, env_, allocator_);
        }

        if (planner_session_) {
            const auto &input_names = planner_session_->get_input_node_names_str();
            if (config_.version == 1) {
                if (input_names.size() != 11) {
                    std::cout << "Model version: 1" << std::endl;
                    std::cout << "Model has " << input_names.size() << " inputs, expected 11" << std::endl;
                    std::cout << "✗ Failed to initialize ONNX engine" << std::endl;
                    throw std::runtime_error("Failed to initialize ONNX engine");
                }
            } else if (config_.version == 0) {
                if (input_names.size() != 6) {
                    std::cout << "Model version: 0" << std::endl;
                    std::cout << "Model has " << input_names.size() << " inputs, expected 6" << std::endl;
                    std::cout << "✗ Failed to initialize ONNX engine" << std::endl;
                    throw std::runtime_error("Failed to initialize ONNX engine");
                }
            } else {
                std::cout << "Model version: " << config_.version << std::endl;
                std::cout << "unsupported model version" << std::endl;
                std::cout << "✗ Failed to initialize ONNX engine" << std::endl;
                throw std::runtime_error("Failed to initialize ONNX engine");
            }

            // Require presence of expected input names
            std::vector<std::string> requiredNames = {
                std::string("context_mujoco_qpos"),
                std::string("target_vel"),
                std::string("mode"),
                std::string("movement_direction"),
                std::string("facing_direction"),
                std::string("random_seed")
            };
            if (config_.version == 1) {
                requiredNames.push_back("height");
                requiredNames.push_back("has_specific_target");
                requiredNames.push_back("specific_target_positions");
                requiredNames.push_back("specific_target_headings");
                requiredNames.push_back("allowed_pred_num_tokens");
            }
            for (const auto &req : requiredNames) {
                if (std::find(input_names.begin(), input_names.end(), req) == input_names.end()) {
                    std::cout << "✗ Missing required input tensor: " << req << std::endl;
                    throw std::runtime_error("Missing required ONNX input tensor: " + req);
                }
            }
        }
    }

    /**
     * @brief Destructor
     */
    ~LocalMotionPlannerONNX() = default;

    bool InitializeSpecific() override {
        if (!planner_session_) {
            return false;
        }
        
        // ==== CLEANUP EXISTING STATE BEFORE INITIALIZATION ====
        std::cout << "Cleaning up existing ONNX planner state..." << std::endl;
        
        // Clear all input value vectors
        mode_values_.clear();
        target_vel_values_.clear();
        movement_direction_values_.clear();
        facing_direction_values_.clear();
        random_seed_values_.clear();
        context_qpos_values_.clear();

        mode_values_.resize(1);
        target_vel_values_.resize(1);
        movement_direction_values_.resize(3, 0.0f);
        facing_direction_values_.resize(3, 0.0f);
        random_seed_values_.resize(1);
        context_qpos_values_.resize(4 * (G1_NUM_MOTOR + 7));
        
        {
            // version 1 extra inputs
            target_height_values_.clear();
            has_specific_target_.clear();
            specific_target_positions_.clear();
            specific_target_headings_.clear();
            allowed_pred_num_tokens_.clear();
            
            target_height_values_.resize(1, -1.0f);
            has_specific_target_.resize(1, 0);
            specific_target_positions_.resize(12, 0.0f);
            specific_target_headings_.resize(4, 0.0f);
            allowed_pred_num_tokens_.resize(11, 0);

            // default allowed tokens: [1,1,1,1,1,1,0,0,0,0,0]
            if (allowed_pred_num_tokens_.size() >= 6) {
                allowed_pred_num_tokens_[0] = 1;
                allowed_pred_num_tokens_[1] = 1;
                allowed_pred_num_tokens_[2] = 1;
                allowed_pred_num_tokens_[3] = 1;
                allowed_pred_num_tokens_[4] = 1;
                allowed_pred_num_tokens_[5] = 1;
            }
        }

        const auto &input_names = planner_session_->get_input_node_names_str();
        const auto &input_dims = planner_session_->get_input_node_dims();
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string &name = input_names[i];
            if (name == "context_mujoco_qpos") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), context_qpos_values_.data(), context_qpos_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "target_vel") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), target_vel_values_.data(), target_vel_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "mode") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<int64_t>(allocator_.GetInfo(), mode_values_.data(), mode_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "movement_direction") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), movement_direction_values_.data(), movement_direction_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "facing_direction") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), facing_direction_values_.data(), facing_direction_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "random_seed") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<int64_t>(allocator_.GetInfo(), random_seed_values_.data(), random_seed_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "height") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), target_height_values_.data(), target_height_values_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "has_specific_target") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<int64_t>(allocator_.GetInfo(), has_specific_target_.data(), has_specific_target_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "specific_target_positions") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), specific_target_positions_.data(), specific_target_positions_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "specific_target_headings") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), specific_target_headings_.data(), specific_target_headings_.size(), input_dims[i].data(), input_dims[i].size()));
            } else if (name == "allowed_pred_num_tokens") {
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<int64_t>(allocator_.GetInfo(), allowed_pred_num_tokens_.data(), allowed_pred_num_tokens_.size(), input_dims[i].data(), input_dims[i].size()));
            } else {
                // Unknown input; create a zero tensor of expected type/shape to satisfy model binding
                // Default to float tensor
                std::vector<float> zero;
                size_t count = 1;
                for (auto d : input_dims[i]) { if (d > 0) count *= static_cast<size_t>(d); }
                zero.resize(count, 0.0f);
                planner_input_tensors_.push_back(Ort::Value::CreateTensor<float>(allocator_.GetInfo(), zero.data(), zero.size(), input_dims[i].data(), input_dims[i].size()));
            }
        }

        planner_output_tensors_.clear();
        
        // Reset data pointers
        mujoco_qpos_data_ = nullptr;
        num_pred_frames_data_ = nullptr;

        return true;
    }

    void RunInference() override {
        // Run ONNX inference with default IDLE parameters
        planner_output_tensors_ = planner_session_->Run(planner_input_tensors_);

        // Store pointers to tensor data for consistent ExtractPlannerData interface
        mujoco_qpos_data_ = planner_output_tensors_[0].GetTensorData<float>();
        num_pred_frames_data_ = planner_output_tensors_[1].GetTensorData<int32_t>();
    }


private:
    // ------------------------------------------------------------------
    // ONNX Runtime components
    // ------------------------------------------------------------------
    std::shared_ptr<OrtSession> planner_session_;  ///< Loaded ONNX model session.
    Ort::Env& env_;                                 ///< Shared ONNX Runtime environment.
    Ort::AllocatorWithDefaultOptions& allocator_;   ///< Shared memory allocator.

    // ------------------------------------------------------------------
    // Input data buffers (aliased by the Ort::Value tensors — zero-copy)
    // ------------------------------------------------------------------
    std::vector<float> context_qpos_values_;          ///< [4 × 36] Context: 4 frames × (7 root + 29 joints).
    std::vector<int64_t> mode_values_;                ///< [1] Locomotion mode (LocomotionMode cast).
    std::vector<float> target_vel_values_;            ///< [1] Target speed (−1 = default).
    std::vector<float> movement_direction_values_;    ///< [3] Movement direction unit vector.
    std::vector<float> facing_direction_values_;      ///< [3] Facing direction unit vector.
    std::vector<int64_t> random_seed_values_;         ///< [1] Random seed for stochastic generation.

    // Version 1+ additional inputs
    std::vector<float> target_height_values_;         ///< [1] Target body height (−1 = default).
    std::vector<int64_t> has_specific_target_;        ///< [1] Whether a specific waypoint target is set.
    std::vector<float> specific_target_positions_;    ///< [12] 4 waypoint positions × xyz.
    std::vector<float> specific_target_headings_;     ///< [4]  4 waypoint heading angles.
    std::vector<int64_t> allowed_pred_num_tokens_;    ///< [11] Allowed prediction token mask.
    std::vector<Ort::Value> planner_input_tensors_;   ///< ONNX input tensor handles (alias above buffers).
    
    // ------------------------------------------------------------------
    // Output tensors
    // ------------------------------------------------------------------
    std::vector<Ort::Value> planner_output_tensors_;  ///< Returned by session.Run().
    const float* mujoco_qpos_data_;                   ///< Pointer into output[0]: predicted qpos [frames × 36].
    const int32_t* num_pred_frames_data_;             ///< Pointer into output[1]: number of predicted frames.
    
    /// Write new locomotion commands into the input buffers (called by base UpdatePlanning).
    void UpdateInputTensors(int mode_value,
                           float target_vel,
                           float target_height,
                           const std::array<float, 3>& movement_direction,
                           const std::array<float, 3>& facing_direction,
                           int random_seed) override{
        // Update mode
        mode_values_[0] = mode_value;
        
        // Update target velocity
        target_vel_values_[0] = target_vel;
        if (config_.version == 1) {
            target_height_values_[0] = target_height;
        }
        
        // Update movement direction
        movement_direction_values_[0] = movement_direction[0];
        movement_direction_values_[1] = movement_direction[1];
        movement_direction_values_[2] = movement_direction[2];
        
        // Update facing direction
        facing_direction_values_[0] = facing_direction[0];
        facing_direction_values_[1] = facing_direction[1];
        facing_direction_values_[2] = facing_direction[2];
        
        // Update random seed if provided
        if (random_seed != -1) {
            current_random_seed_ = random_seed;
            random_seed_values_[0] = random_seed;
        }
        
        // Log replanning values
        std::cout << "Replanning with mode: ";
        switch(mode_values_[0])
        {
            case 0:
                std::cout << "IDLE";
                break;
            case 1:
                std::cout << "SLOW_WALK";
                break;
            case 2:
                std::cout << "WALK";
                break;
            case 3:
                std::cout << "RUN";
                break;
            case 4:
                std::cout << "BOXING";
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }
        if (config_.version == 1) {
            std::cout << ", target_height: " << target_height_values_[0];
        }
        std::cout << ", target_vel: " << target_vel_values_[0]
                  << ", movement: [" << movement_direction_values_[0] << ", " << movement_direction_values_[1] << ", " << movement_direction_values_[2] << "]"
                  << ", facing: [" << facing_direction_values_[0] << ", " << facing_direction_values_[1] << ", " << facing_direction_values_[2] << "]" << std::endl;
    }

    virtual float *GetContextBuffer() override {
        return context_qpos_values_.data();
    }
    virtual int32_t GetNumPredFrames() override {
        return num_pred_frames_data_[0];
    }   
    virtual const float *GetMujocoQposBuffer() override {
        return mujoco_qpos_data_;
    }
    virtual float *GetMovementDirectionValues() override {
        return movement_direction_values_.data();
    }
    virtual float *GetFacingDirectionValues() override {
        return facing_direction_values_.data();
    }
    virtual float GetTargetVelValue() override {
        return target_vel_values_[0];
    }
    virtual float GetHeightValue() override {
        return target_height_values_[0];
    }
    virtual int32_t GetRandomSeedValue() override {
        return random_seed_values_[0];
    }
    virtual int32_t GetModeValue() override {
        return mode_values_[0];
    }

};

#endif // LOCALMOTION_KPLANNER_ONNX_HPP
