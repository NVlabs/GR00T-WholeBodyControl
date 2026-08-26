/**
 * @file control_policy.hpp
 * @brief TensorRT-accelerated control-policy engine (observations → actions).
 *
 * PolicyEngine loads an ONNX policy model, converts it to TensorRT at first
 * run (cached on disk), and provides GPU-accelerated inference with optional
 * CUDA graph capture for deterministic, sub-millisecond latency.
 *
 * ## I/O Contract
 *
 *   - **Input**: single tensor `obs_dict` (float32, dimension = policy obs size).
 *   - **Output**: single tensor `action` (float32, dimension = G1_NUM_MOTOR = 29).
 *
 * ## Typical Usage
 *
 *   1. `Initialize(model_path)` – convert + load the TRT engine.
 *   2. Fill `GetInputBuffer()` with observation data.
 *   3. `Infer()` – runs GPU inference and populates `GetActionBuffer()`.
 *   4. (Optional) `CaptureGraph()` after the first successful `Infer()` to
 *      lock the CUDA graph for all subsequent calls.
 */

#ifndef POLICY_ENGINE_HPP
#define POLICY_ENGINE_HPP

#include <memory>
#include <string>
#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <TRTInference/InferenceEngine.h>
#include "robot_parameters.hpp"

/**
 * @class PolicyEngine
 * @brief Runs the main RL control policy on GPU via TensorRT.
 *
 * Owns CUDA resources (stream, graph, graph-exec) and pinned-memory I/O
 * buffers.  Non-copyable; destroyed via `Destroy()` or the destructor.
 */
class PolicyEngine {
public:
  PolicyEngine() = default;
  ~PolicyEngine() { Destroy(); }

  /**
   * @brief Initialize the control policy from a model path
   * @param model_path Path to ONNX model file (will be converted to TensorRT)
   * @param use_fp16 Whether to use FP16 precision
   * @return true if initialization successful, false otherwise
   */
  bool Initialize(const std::string& model_path, bool use_fp16 = false) {
    if (model_path.empty()) {
      std::cerr << "✗ PolicyEngine::Initialize - Empty model path" << std::endl;
      return false;
    }

    config_.model_path = model_path;
    config_.use_fp16 = use_fp16;

    try {
      std::cout << "Loading policy model..." << std::endl;

      inference_engine_ = std::make_unique<TRTInferenceEngine>();

      // Setup options for ONNX to TensorRT conversion
      Options options;
      options.deviceID = config_.device_id;
      std::string prefix("policy_");
      if (use_fp16) { 
        options.precision = Precision::FP16; 
        prefix += "fp16_"; 
      }

      std::string cached_trt_file;
      if (!ConvertONNXToTRT(options, model_path, cached_trt_file, prefix, false)) {
        std::cerr << "✗ Failed to convert policy ONNX to TRT: " << model_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->Initialize(cached_trt_file, options.deviceID, options.dynamic_axes_names)) {
        std::cerr << "✗ Failed to initialize policy TensorRT model: " << cached_trt_file << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->InitInputs({})) {
        std::cerr << "✗ Failed to initialize policy TensorRT model inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }

      std::cout << "✓ Successfully converted ONNX to TRT: " << cached_trt_file << std::endl;

      // Validate required inputs
      auto input_names = inference_engine_->GetInputTensorNames();
      if (input_names.size() != 1) {
        std::cerr << "✗ Policy must have exactly 1 input, found " << input_names.size() << ": ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(input_names.begin(), input_names.end(), std::string("obs_dict")) == input_names.end()) {
        std::cerr << "✗ Policy input tensor 'obs_dict' not found. Available inputs: ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      input_tensor_name_ = "obs_dict";
      if (inference_engine_->GetTensorDataType(input_tensor_name_) != DataType::FLOAT) {
        std::cerr << "✗ Policy input 'obs_dict' must be float32" << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize input buffer
      std::vector<int64_t> input_dims;
      inference_engine_->GetTensorShape(input_tensor_name_, input_dims);
      
      config_.input_dimension = std::accumulate(
        input_dims.begin(), input_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>()
      );
      policy_input_buffer_.resize(config_.input_dimension, 0.0f);
      
      // Set initial input data (zeros)
      inference_engine_->SetInputData(input_tensor_name_, policy_input_buffer_);

      // Validate required outputs
      auto output_names = inference_engine_->GetOutputTensorNames();
      if (output_names.empty()) {
        std::cerr << "✗ Policy model has no outputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (output_names.size() != 1) {
        std::cerr << "✗ Policy must have exactly 1 output, found " << output_names.size() << ": ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(output_names.begin(), output_names.end(), std::string("action")) == output_names.end()) {
        std::cerr << "✗ Policy output tensor 'action' not found. Available outputs: ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      output_tensor_name_ = "action";

      // Get output dimensions
      std::vector<int64_t> output_dims;
      if (inference_engine_->GetTensorShape(output_tensor_name_, output_dims)) {
        config_.action_dimension = std::accumulate(
          output_dims.begin(), output_dims.end(), 1, std::multiplies<size_t>()
        );
      }

      // Validate action dimension matches robot configuration
      if (config_.action_dimension != G1_NUM_MOTOR) {
        std::cerr << "✗ Policy action dimension (" << config_.action_dimension 
                  << ") doesn't match G1 robot motors (" << G1_NUM_MOTOR << ")" << std::endl;
        inference_engine_.reset();
        return false;
      }

      action_buffer_.resize(config_.action_dimension, 0.0f);

      // Create CUDA stream
      cudaError_t cuda_status = cudaStreamCreate(&cuda_stream_);
      if (cuda_status != cudaSuccess) {
        std::cerr << "✗ Failed to create CUDA stream: " << cudaGetErrorString(cuda_status) << std::endl;
        inference_engine_.reset();
        return false;
      }

      initialized_ = true;
      std::cout << "✓ Policy engine initialized successfully!" << std::endl;
      std::cout << "  Model: " << model_path << std::endl;
      std::cout << "  Input dimension: " << config_.input_dimension << std::endl;
      std::cout << "  Action dimension: " << config_.action_dimension << std::endl;
      std::cout << "  Input tensor: " << input_tensor_name_ << std::endl;
      std::cout << "  Output tensor: " << output_tensor_name_ << std::endl;
      std::cout << "  Precision: " << (use_fp16 ? "FP16" : "FP32") << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "✗ PolicyEngine::Initialize - Exception: " << e.what() << std::endl;
      inference_engine_.reset();
      return false;
    }
  }

  /**
   * @brief Set input data for the control policy
   * @param data Input data buffer
   * @param element_count Number of elements
   */
  template<typename T>
  void SetInputData(const T* data, size_t element_count) {
    if (!initialized_ || !inference_engine_) { return; }
    inference_engine_->SetInputData(input_tensor_name_, data, element_count);
  }

  /**
   * @brief Set input data asynchronously
   * @param data Input data buffer
   * @param element_count Number of elements
   * @param stream CUDA stream for async operation
   */
  template<typename T>
  void SetInputDataAsync(const T* data, size_t element_count, cudaStream_t stream) {
    if (!initialized_ || !inference_engine_) { return; }
    inference_engine_->SetInputDataAsync(input_tensor_name_, data, element_count, stream);
  }

  /**
   * @brief Set input data using TPinnedVector
   * @param data Input data in TPinnedVector
   */
  template<typename T>
  void SetInputData(const TPinnedVector<T>& data) {
    if (!initialized_ || !inference_engine_) { return; }
    inference_engine_->SetInputData(input_tensor_name_, data);
  }

  /**
   * @brief Run control policy inference and populate internal action buffer
   * @param stream CUDA stream for inference (uses internal stream if nullptr)
   * @return true if inference successful, false otherwise
   */
  bool Infer(cudaStream_t stream = nullptr) {
    if (!initialized_) { 
      std::cerr << "✗ PolicyEngine::Infer - Not initialized" << std::endl; 
      return false; 
    }
    if (!inference_engine_) { 
      std::cerr << "✗ PolicyEngine::Infer - TensorRT engine not initialized" << std::endl; 
      return false; 
    }

    cudaStream_t infer_stream = (stream != nullptr) ? stream : cuda_stream_;

    // Transfer input data from CPU to GPU
    inference_engine_->SetInputData(input_tensor_name_, policy_input_buffer_);

    if (graph_captured_ && cuda_graph_exec_ != nullptr) {
      cudaError_t status = cudaGraphLaunch(cuda_graph_exec_, infer_stream);
      if (status != cudaSuccess) {
        std::cerr << "✗ PolicyEngine::Infer - Failed to launch CUDA graph: " << cudaGetErrorString(status) << std::endl;
        return false;
      }
    } else {
      if (!inference_engine_->Enqueue(infer_stream)) {
        std::cerr << "✗ PolicyEngine::Infer - Failed to enqueue inference" << std::endl;
        return false;
      }
    }
    
    // Automatically populate internal action buffer after inference (GPU to CPU)
    inference_engine_->GetOutputDataAsync(output_tensor_name_, action_buffer_, infer_stream);
    cudaStreamSynchronize(infer_stream);
    
    return true;
  }

  /**
   * @brief Capture CUDA graph for optimized execution
   * @return true if capture successful, false otherwise
   */
  bool CaptureGraph() {
    if (!initialized_ || !inference_engine_) {
      std::cerr << "✗ PolicyEngine::CaptureGraph - Cannot capture (engine not initialized)" << std::endl;
      return false;
    }
    if (graph_captured_) { 
      std::cout << "Control policy CUDA graph already captured" << std::endl;
      return true; 
    }

    std::cout << "Capturing control policy CUDA graph..." << std::endl;
    cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed);
    if (!inference_engine_->Enqueue(cuda_stream_)) {
      std::cerr << "✗ Failed to enqueue control policy inference for CUDA graph capture" << std::endl;
      cudaStreamEndCapture(cuda_stream_, &cuda_graph_);
      return false;
    }
    cudaStreamEndCapture(cuda_stream_, &cuda_graph_);
    cudaStreamSynchronize(cuda_stream_);
    
    cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, NULL, NULL, 0);
    
    graph_captured_ = true;
    std::cout << "✓ Control policy CUDA graph captured successfully!" << std::endl;
    return true;
  }

  /**
   * @brief Get CUDA stream used by control policy
   * @return CUDA stream handle
   */
  cudaStream_t GetCudaStream() const { return cuda_stream_; }

  /**
   * @brief Get CUDA graph (if captured)
   * @return CUDA graph handle
   */
  cudaGraph_t GetCudaGraph() const { return cuda_graph_; }

  /**
   * @brief Get CUDA graph execution instance
   * @return CUDA graph execution handle
   */
  cudaGraphExec_t GetCudaGraphExec() const { return cuda_graph_exec_; }

  /**
   * @brief Get input dimension
   * @return Input dimension size
   */
  size_t GetInputDimension() const { return config_.input_dimension; }

  /**
   * @brief Get action dimension
   * @return Action dimension size
   */
  size_t GetActionDimension() const { return config_.action_dimension; }

  /**
   * @brief Check if control policy is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const { return initialized_; }

  /**
   * @brief Get input tensor names
   * @return Vector of input tensor names
   */
  std::vector<std::string> GetInputTensorNames() const {
    if (!initialized_ || !inference_engine_) { return {}; }
    return inference_engine_->GetInputTensorNames();
  }

  /**
   * @brief Get output tensor names
   * @return Vector of output tensor names
   */
  std::vector<std::string> GetOutputTensorNames() const {
    if (!initialized_ || !inference_engine_) { return {}; }
    return inference_engine_->GetOutputTensorNames();
  }

  /**
   * @brief Get reference to internal input buffer
   * @return Reference to input buffer
   */
  TPinnedVector<float>& GetInputBuffer() { return policy_input_buffer_; }

  /**
   * @brief Get reference to internal action buffer
   * @return Reference to action buffer
   */
  TPinnedVector<float>& GetActionBuffer() { return action_buffer_; }

  /**
   * @brief Get input tensor name
   * @return Input tensor name
   */
  const std::string& GetInputTensorName() const { return input_tensor_name_; }

  /**
   * @brief Get output tensor name
   * @return Output tensor name
   */
  const std::string& GetOutputTensorName() const { return output_tensor_name_; }

  /**
   * @brief Destroy and clean up control policy resources
   */
  void Destroy() {
    if (!initialized_) { return; }
    if (cuda_graph_exec_ != nullptr) { 
      cudaGraphExecDestroy(cuda_graph_exec_); 
      cuda_graph_exec_ = nullptr; 
    }
    if (cuda_graph_ != nullptr) { 
      cudaGraphDestroy(cuda_graph_); 
      cuda_graph_ = nullptr; 
    }
    if (cuda_stream_ != nullptr) { 
      cudaStreamDestroy(cuda_stream_); 
      cuda_stream_ = nullptr; 
    }
    if (inference_engine_) { 
      inference_engine_->Destroy(); 
      inference_engine_.reset(); 
    }
    graph_captured_ = false; 
    initialized_ = false;
  }

private:
  // Internal configuration state
  struct Config {
    std::string model_path;
    int device_id = 0;
    size_t input_dimension = 0;
    size_t action_dimension = 0;
    bool use_fp16 = false;
  };
  Config config_;

  // TensorRT inference engine
  std::unique_ptr<TRTInferenceEngine> inference_engine_;

  // Tensor names
  std::string input_tensor_name_;
  std::string output_tensor_name_;

  // CUDA resources
  cudaStream_t cuda_stream_ = nullptr;
  cudaGraph_t cuda_graph_ = nullptr;
  cudaGraphExec_t cuda_graph_exec_ = nullptr;

  // Input buffer
  TPinnedVector<float> policy_input_buffer_;

  // Action output buffer
  TPinnedVector<float> action_buffer_;

  // State
  bool initialized_ = false;
  bool graph_captured_ = false;
};

#endif // POLICY_ENGINE_HPP

