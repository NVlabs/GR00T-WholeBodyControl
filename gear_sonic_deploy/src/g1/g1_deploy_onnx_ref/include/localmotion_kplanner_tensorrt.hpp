/**
 * @file localmotion_kplanner_tensorrt.hpp
 * @brief TensorRT (GPU) backend for the locomotion planner.
 *
 * LocalMotionPlannerTensorRT is a concrete implementation of
 * LocalMotionPlannerBase that runs planner inference on the GPU via NVIDIA
 * TensorRT.  It is the production backend, offering significantly lower
 * latency than the ONNX Runtime backend.
 *
 * ## GPU Pipeline
 *
 *   1. Constructor converts the ONNX model to a TensorRT engine (cached on
 *      disk), creates a CUDA stream, and captures a CUDA graph for the
 *      inference pass.
 *   2. `UpdateInputTensors()` writes new locomotion commands into pinned-
 *      memory (`TPinnedVector`) buffers on the host.
 *   3. `RunInference()` asynchronously copies inputs to the GPU, launches
 *      the captured CUDA graph, copies outputs back, and synchronises.
 *   4. Output buffers (`mujoco_qpos_values_`, `num_pred_frames_values_`)
 *      are read by the base class to resample the trajectory at 50 Hz.
 *
 * ## Model Versions
 *
 *   Version | Inputs | Notes
 *   --------|--------|------
 *   0       | 6      | Basic: context, mode, target_vel, movement/facing direction, random_seed.
 *   1–2     | 11     | Adds: height, has_specific_target, specific_target_positions/headings, allowed_pred_num_tokens.
 *
 * ## CUDA Graph
 *
 * A CUDA graph is captured during `InitializeEngine()` and replayed on every
 * `RunInference()` call.  This eliminates kernel launch overhead and ensures
 * deterministic timing (~1–2 ms per planning cycle on Jetson Orin).
 */

#ifndef LOCALMOTION_KPLANNER_TENSORRT_HPP
#define LOCALMOTION_KPLANNER_TENSORRT_HPP

#include <TRTInference/InferenceEngine.h>
#include <iostream>
#include "localmotion_kplanner.hpp"
#include <cuda_runtime.h>

/**
 * @class LocalMotionPlannerTensorRT
 * @brief TensorRT (GPU) backend for the locomotion planner.
 *
 * Uses CUDA graphs and pinned memory for low-latency, deterministic inference.
 */
class LocalMotionPlannerTensorRT : public LocalMotionPlannerBase {
public:
    /**
     * @brief Constructor for LocalMotionPlannerTensorRT
     * @param use_fp16 Whether to use FP16 precision
     * @param device_id CUDA device ID to use for inference
     * @param config Planner configuration parameters
     */
    LocalMotionPlannerTensorRT(bool use_fp16,
                               int device_id = 0,
                               const PlannerConfig& config = PlannerConfig())
        : LocalMotionPlannerBase(config), use_fp16_(use_fp16),
          device_id_(device_id), cuda_stream_(nullptr), graph_(nullptr), graphExec_(nullptr) {
        
        // Initialize TensorRT engine
        inference_engine_ = std::make_unique<TRTInferenceEngine>();

        // Clear all input value vectors and ensure correct sizes:
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
            // version 1
            target_height_values_.clear();
            has_specific_target_.clear(); 
            specific_target_positions_.clear(); 
            specific_target_headings_.clear(); 
            allowed_pred_num_tokens_.clear(); 

            target_height_values_.resize(1,-1.0f); 
            has_specific_target_.resize(1,0); 
            specific_target_positions_.resize(12,0.0f); 
            specific_target_headings_.resize(4,0.0f); 
            allowed_pred_num_tokens_.resize(11,0);

            // set allowed_pred_num_tokens_ to [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            allowed_pred_num_tokens_[0] = 0; // 6 tokens
            allowed_pred_num_tokens_[1] = 0; // 7
            allowed_pred_num_tokens_[2] = 0; // 8
            allowed_pred_num_tokens_[3] = 1; // 9
            allowed_pred_num_tokens_[4] = 1; // 10
            allowed_pred_num_tokens_[5] = 1;
        }


        // Clear output buffers
        mujoco_qpos_values_.clear();
        num_pred_frames_values_.clear();
        
        // Initialize output vectors with defaults
        mujoco_qpos_values_.resize(64 * 36);
        num_pred_frames_values_.resize(1);

        bool success = InitializeEngine();
        if (!success) {
            std::cout << "✗ Failed to initialize TensorRT engine" << std::endl;
            throw std::runtime_error("Failed to initialize TensorRT engine");
        }
        
        // CUDA stream will be created in the .cpp implementation
        // to avoid including full CUDA headers in the header file
    }

    /**
     * @brief Destructor
     */
    ~LocalMotionPlannerTensorRT() {
        if (graphExec_) {
            cudaGraphExecDestroy(graphExec_);
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
        if (cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
        }
    }

    bool InitializeSpecific() override {
        if (!inference_engine_) {
            return false;
        }
        

        return true;
    }


    // TensorRT always uses CUDA streams for async inference (like existing code)

    /**
     * @brief Get current CUDA stream (for async operations)
     * @return Current CUDA stream
     */
    cudaStream_t GetCudaStream() const { return cuda_stream_; }

private:
    // ------------------------------------------------------------------
    // TensorRT engine components
    // ------------------------------------------------------------------
    std::unique_ptr<TRTInferenceEngine> inference_engine_;  ///< TensorRT inference engine.
    bool use_fp16_;           ///< FP16 precision flag.
    int device_id_;           ///< CUDA device ID.
    cudaStream_t cuda_stream_;  ///< CUDA stream for async operations.
    cudaGraph_t graph_;         ///< Captured CUDA graph (inference pass).
    cudaGraphExec_t graphExec_; ///< Instantiated CUDA graph for replay.

    // ------------------------------------------------------------------
    // Input data buffers (pinned memory for efficient CPU ↔ GPU transfers)
    // ------------------------------------------------------------------
    TPinnedVector<float> context_qpos_values_;          ///< [4 × 36] Context frames (7 root + 29 joints each).
    TPinnedVector<int64_t> mode_values_;                ///< [1] Locomotion mode.
    TPinnedVector<float> target_vel_values_;            ///< [1] Target speed (−1 = default).
    TPinnedVector<float> movement_direction_values_;    ///< [3] Movement direction unit vector.
    TPinnedVector<float> facing_direction_values_;      ///< [3] Facing direction unit vector.
    TPinnedVector<int64_t> random_seed_values_;         ///< [1] Random seed.

    // Version 1–2 additional inputs
    TPinnedVector<float> target_height_values_;         ///< [1] Target body height (−1 = default).
    TPinnedVector<int64_t> has_specific_target_;        ///< [1] Whether specific waypoint target is set.
    TPinnedVector<float> specific_target_positions_;    ///< [12] 4 waypoint positions × xyz.
    TPinnedVector<float> specific_target_headings_;     ///< [4]  4 waypoint heading angles.
    TPinnedVector<int64_t> allowed_pred_num_tokens_;    ///< [11] Allowed prediction token mask.

    // ------------------------------------------------------------------
    // Output data buffers (pinned memory)
    // ------------------------------------------------------------------
    TPinnedVector<float> mujoco_qpos_values_;           ///< [frames × 36] Predicted qpos (7 root + 29 joints).
    TPinnedVector<int32_t> num_pred_frames_values_;     ///< [1] Number of predicted 30 Hz frames.

    /// Named tensor identifiers used for TensorRT SetInput/GetOutput calls.
    struct TensorNames {
        std::string context_qpos = "context_mujoco_qpos";
        std::string mode = "mode";
        std::string target_vel = "target_vel";
        std::string target_height = "height"; // version 1
        std::string has_specific_target = "has_specific_target"; // version 1
        std::string specific_target_positions = "specific_target_positions"; // version 1
        std::string specific_target_headings = "specific_target_headings"; // version 1
        std::string allowed_pred_num_tokens = "allowed_pred_num_tokens"; // version 1

        std::string movement_direction = "movement_direction";
        std::string facing_direction = "facing_direction";
        std::string random_seed = "random_seed";
        
        std::string mujoco_qpos_output = "mujoco_qpos";
        std::string num_pred_frames_output = "num_pred_frames";
    } tensor_names_;

    /**
     * @brief Convert the ONNX model to TensorRT, initialise the engine, and capture a CUDA graph.
     * @return True on success; false if conversion, initialisation, or validation fails.
     */
    bool InitializeEngine() {
        std::cout << "Initialize Engine..." << std::endl;

        // Create CUDA stream for async operations
        cudaError_t status = cudaStreamCreate(&cuda_stream_);
        if (status != cudaSuccess) {
            std::cout << "✗ CUDA error: " << cudaGetErrorString(status) << std::endl;
            return false;
        }

        // Setup options for ONNX to TensorRT conversion
        Options options;
        options.deviceID = device_id_;
        std::string prefix("planner_");
        if(use_fp16_)
        {
            options.precision = Precision::FP16;
            prefix += "fp16_";
        }
        
        std::string cachedTRTFile;
        std::string onnxModelPath = config_.model_path;
        
        // Convert ONNX to TensorRT if needed
        if (!ConvertONNXToTRT(options, onnxModelPath, cachedTRTFile, prefix, false)) {
            std::cout << "✗ Failed to convert ONNX at " << onnxModelPath << std::endl;
            return false;
        }
        
        // Initialize the TensorRT inference engine
        if (!inference_engine_->Initialize(cachedTRTFile, options.deviceID, options.dynamic_axes_names)) {
            std::cout << "✗ Failed to initialize TensorRT model: " << cachedTRTFile << std::endl;
            return false;
        }
        
        // Initialize engine inputs
        if (!inference_engine_->InitInputs({})) {
            std::cout << "✗ Failed to initialize TensorRT model inputs: " << cachedTRTFile << std::endl;
            return false;
        }
        
        std::cout << "✓ Successfully converted ONNX to TRT: " << cachedTRTFile << std::endl;

        // Capture a CUDA graph
        cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed);
        if(!inference_engine_->Enqueue(cuda_stream_))
        {
            std::cout << "✗ Failed to enqueue inference for capturing cuda graph" << std::endl;
            return false;
        }
        cudaStreamEndCapture(cuda_stream_, &graph_);
        cudaStreamSynchronize(cuda_stream_);
        
        cudaGraphInstantiate(&graphExec_, graph_, NULL, NULL, 0);

        // Log tensor information for debugging
        std::vector<std::string> inputNames = inference_engine_->GetInputTensorNames();
        std::vector<std::string> outputNames = inference_engine_->GetOutputTensorNames();

        // Version-based input validation (ensure expected count and required names)
        size_t expectedInputs = (config_.version == 1 || config_.version == 2) ? 11 : 6;
        if (inputNames.size() != expectedInputs) {
            std::cout << "Model version: " << config_.version << std::endl;
            std::cout << "Model has " << inputNames.size() << " inputs, expected " << expectedInputs << std::endl;
            std::cout << "✗ Failed to initialize TensorRT engine (input count mismatch)" << std::endl;
            return false;
        }

        {
            std::vector<std::string> requiredNames = {
                tensor_names_.context_qpos,
                tensor_names_.target_vel,
                tensor_names_.mode,
                tensor_names_.movement_direction,
                tensor_names_.facing_direction,
                tensor_names_.random_seed
            };
            if (config_.version == 1 || config_.version == 2) {
                requiredNames.push_back(tensor_names_.target_height);
                requiredNames.push_back(tensor_names_.has_specific_target);
                requiredNames.push_back(tensor_names_.specific_target_positions);
                requiredNames.push_back(tensor_names_.specific_target_headings);
                requiredNames.push_back(tensor_names_.allowed_pred_num_tokens);
            }
            for (const auto &req : requiredNames) {
                if (std::find(inputNames.begin(), inputNames.end(), req) == inputNames.end()) {
                    std::cout << "✗ Missing required input tensor: " << req << std::endl;
                    return false;
                }
            }
        }
        
        std::cout << "Input tensors:" << std::endl;
        for (const auto& name : inputNames) {
            std::vector<int64_t> shape;
            inference_engine_->GetTensorShape(name, shape);
            std::cout << "  " << name << " ";
            for (auto& dim : shape) {
                std::cout << dim << " ";
            }
            
            auto dataType = inference_engine_->GetTensorDataType(name);
            if (dataType == DataType::FLOAT) {
                std::cout << "float";
            } else if (dataType == DataType::INT64) {
                std::cout << "int64";
            } else if (dataType == DataType::INT32) {
                std::cout << "int32";
            } else {
                std::cout << "unknown";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Output tensors:" << std::endl;
        for (const auto& name : outputNames) {
            std::vector<int64_t> shape;
            inference_engine_->GetTensorShape(name, shape);
            std::cout << "  " << name << " ";
            for (auto& dim : shape) {
                std::cout << dim << " ";
            }
            
            auto dataType = inference_engine_->GetTensorDataType(name);
            if (dataType == DataType::FLOAT) {
                std::cout << "float";
            } else if (dataType == DataType::INT32) {
                std::cout << "int32";
            } else {
                std::cout << "unknown";
            }
            std::cout << std::endl;
        }

        std::cout << "✓ TensorRT planner model loaded successfully!" << std::endl;
        return true;
    }
    
    /// Async GPU inference: copy inputs → launch CUDA graph → copy outputs → synchronise.
    void RunInference() override {
        inference_engine_->SetInputDataAsync(tensor_names_.context_qpos, context_qpos_values_, cuda_stream_);
        inference_engine_->SetInputDataAsync(tensor_names_.facing_direction, facing_direction_values_, cuda_stream_);
        inference_engine_->SetInputDataAsync(tensor_names_.mode, mode_values_, cuda_stream_);
        inference_engine_->SetInputDataAsync(tensor_names_.target_vel, target_vel_values_, cuda_stream_);
        inference_engine_->SetInputDataAsync(tensor_names_.movement_direction, movement_direction_values_, cuda_stream_);
        inference_engine_->SetInputDataAsync(tensor_names_.random_seed, random_seed_values_, cuda_stream_);
        
        if (config_.version == 1 || config_.version == 2)
        {
            inference_engine_->SetInputDataAsync(tensor_names_.target_height, target_height_values_, cuda_stream_);
            inference_engine_->SetInputDataAsync(tensor_names_.has_specific_target, has_specific_target_, cuda_stream_);
            inference_engine_->SetInputDataAsync(tensor_names_.specific_target_positions, specific_target_positions_, cuda_stream_);
            inference_engine_->SetInputDataAsync(tensor_names_.specific_target_headings, specific_target_headings_, cuda_stream_);
            inference_engine_->SetInputDataAsync(tensor_names_.allowed_pred_num_tokens, allowed_pred_num_tokens_, cuda_stream_);
        }
        
        cudaGraphLaunch(graphExec_, cuda_stream_);
        inference_engine_->GetOutputDataAsync(tensor_names_.mujoco_qpos_output, mujoco_qpos_values_, cuda_stream_);
        inference_engine_->GetOutputDataAsync(tensor_names_.num_pred_frames_output, num_pred_frames_values_, cuda_stream_);
        cudaStreamSynchronize(cuda_stream_);
    }
    
    /// Write new locomotion commands into pinned-memory input buffers.
    void UpdateInputTensors(int mode_value,
                           float target_vel,
                           float target_height,
                           const std::array<float, 3>& movement_direction,
                           const std::array<float, 3>& facing_direction,
                           int random_seed) override {
        // Update mode
        if (mode_value < 0 || mode_value >= GetValidModeValueRange()) {
            std::cout << "✗ Invalid mode value: " << mode_value << ". Please verify the planner model version and the mode value range" << std::endl;
        }
        mode_values_[0] = mode_value < GetValidModeValueRange() ? mode_value : 0;

        // Update target velocity
        target_vel_values_[0] = target_vel;

        if (config_.version == 1 || config_.version == 2)
        {
            // Update target height (version 1)
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
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_SQUAT";
                }
                else
                {
                    std::cout << "BOXING";
                }
                break;
            case 5:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_KNEEL_TWO_LEGS";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 6:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_KNEEL";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 7:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_LYING_FACE_DOWN";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 8:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_CRAWLING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 9:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_BOXING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 10:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "WALK_BOXING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 11:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "LEFT_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 12:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RIGHT_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 13:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RANDOM_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 14:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "ELBOW_CRAWLING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 15:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "LEFT_HOOK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 16:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RIGHT_HOOK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 17:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "FORWARD_JUMP";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 18:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "STEALTH_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 19:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "INJURED_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 20:
                if (config_.version == 2)
                {
                    std::cout << "LEDGE_WALKING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 21:
                if (config_.version == 2)
                {
                    std::cout << "OBJECT_CARRYING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 22:
                if (config_.version == 2)
                {
                    std::cout << "STEALTH_WALK_2";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 23:
                if (config_.version == 2)
                {
                    std::cout << "HAPPY_DANCE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 24:
                if (config_.version == 2)
                {
                    std::cout << "ZOMBIE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 25:
                if (config_.version == 2)
                {
                    std::cout << "GUN_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 26:
                if (config_.version == 2)
                {
                    std::cout << "SCARE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }
        if (config_.version == 1 || config_.version == 2)
        {
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
        return num_pred_frames_values_[0];
    }   
    virtual const float *GetMujocoQposBuffer() override {
        return mujoco_qpos_values_.data();
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

#endif // LOCALMOTION_KPLANNER_TENSORRT_HPP
