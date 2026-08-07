/**
 * @file encoder.hpp
 * @brief TensorRT-accelerated encoder engine (observations → token state).
 *
 * EncoderEngine loads an ONNX encoder model, converts it to TensorRT, and
 * provides GPU-accelerated inference with optional CUDA graph capture.
 *
 * The encoder compresses high-dimensional robot observations into a compact
 * latent / token representation that is then consumed by the control policy
 * as the `token_state` observation.
 *
 * ## I/O Contract
 *
 *   - **Input**: single tensor `obs_dict` (float32, encoder observation dim).
 *   - **Output**: single tensor `encoded_tokens` (float32, token dimension).
 *
 * ## Configuration
 *
 * Driven by the `encoder:` section of `observation_config.yaml` (parsed by
 * ObservationConfigParser).  The encoder is created only when `token_state`
 * is enabled in the observation config.  The model path is provided via the
 * `--encoder-model` command-line argument.
 *
 * ## Typical Usage
 *
 *   1. `Initialize(model_path)` – convert + load TRT engine.
 *   2. Fill `GetInputBuffer()` with encoder observations.
 *   3. `Encode()` – runs GPU inference and populates `GetTokenBuffer()`.
 *   4. (Optional) `CaptureGraph()` for deterministic latency.
 */

#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <memory>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <TRTInference/InferenceEngine.h>

/**
 * @class EncoderEngine
 * @brief Runs the observation encoder on GPU via TensorRT.
 *
 * Mirrors the PolicyEngine architecture: owns CUDA resources, pinned-memory
 * buffers, and supports CUDA graph capture.  Non-copyable.
 */
class EncoderEngine {
public:
  EncoderEngine() = default;
  ~EncoderEngine() { Destroy(); }

  /**
   * @brief Initialize the encoder engine from a model path
   * @param model_path Path to ONNX model file (will be converted to TensorRT)
   * @param use_fp16 Whether to use FP16 precision
   * @return true if initialization successful, false otherwise
   */
  bool Initialize(const std::string& model_path, bool use_fp16 = false) {
    if (model_path.empty()) {
      std::cerr << "✗ EncoderEngine::Initialize - Empty model path" << std::endl;
      return false;
    }

    config_.model_path = model_path;
    config_.use_fp16 = use_fp16;

    try {
      std::cout << "Loading encoder model..." << std::endl;

      inference_engine_ = std::make_unique<TRTInferenceEngine>();

      Options options;
      options.deviceID = config_.device_id;
      std::string prefix("encoder_");
      if (use_fp16) { options.precision = Precision::FP16; prefix += "fp16_"; }

      std::string cached_trt_file;
      if (!ConvertONNXToTRT(options, model_path, cached_trt_file, prefix, false)) {
        std::cerr << "✗ Failed to convert encoder ONNX to TRT: " << model_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->Initialize(cached_trt_file, options.deviceID, options.dynamic_axes_names)) {
        std::cerr << "✗ Failed to initialize encoder TensorRT model: " << cached_trt_file << std::endl;
        inference_engine_.reset();
        return false;
      }

      if (!inference_engine_->InitInputs({})) {
        std::cerr << "✗ Failed to initialize encoder TensorRT model inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }

      auto output_names = inference_engine_->GetOutputTensorNames();
      if (output_names.empty()) {
        std::cerr << "✗ Encoder model has no outputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (output_names.size() != 1) {
        std::cerr << "✗ Encoder must have exactly 1 output, found " << output_names.size() << ": ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(output_names.begin(), output_names.end(), std::string("encoded_tokens")) == output_names.end()) {
        std::cerr << "✗ Encoder output tensor 'encoded_tokens' not found. Available outputs: ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      output_tensor_name_ = "encoded_tokens";

      std::vector<int64_t> output_dims;
      if (inference_engine_->GetTensorShape(output_tensor_name_, output_dims)) {
        config_.token_dimension = std::accumulate(output_dims.begin(), output_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
      }

      token_buffer_.resize(config_.token_dimension, 0.0f);

      auto input_names = inference_engine_->GetInputTensorNames();
      if (input_names.empty()) {
        std::cerr << "✗ Encoder model has no inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (input_names.size() != 1) {
        std::cerr << "✗ Encoder must have exactly 1 input, found " << input_names.size() << ": ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(input_names.begin(), input_names.end(), std::string("obs_dict")) == input_names.end()) {
        std::cerr << "✗ Encoder input tensor 'obs_dict' not found. Available inputs: ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      input_tensor_name_ = "obs_dict";
      if (inference_engine_->GetTensorDataType(input_tensor_name_) != DataType::FLOAT) {
        std::cerr << "✗ Encoder input 'obs_dict' must be float32" << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize input buffer
      std::vector<int64_t> input_dims;
      if (!inference_engine_->GetTensorShape(input_tensor_name_, input_dims)) {
        std::cerr << "✗ Failed to get encoder input shape" << std::endl;
        inference_engine_.reset();
        return false;
      }
      
      config_.input_dimension = std::accumulate(
        input_dims.begin(), input_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>()
      );
      encoder_input_buffer_.resize(config_.input_dimension, 0.0f);
      
      // Set initial input data (zeros)
      inference_engine_->SetInputData(input_tensor_name_, encoder_input_buffer_);

      cudaError_t cuda_status = cudaStreamCreate(&cuda_stream_);
      if (cuda_status != cudaSuccess) {
        std::cerr << "✗ Failed to create CUDA stream: " << cudaGetErrorString(cuda_status) << std::endl;
        inference_engine_.reset();
        return false;
      }

      initialized_ = true;
      std::cout << "✓ Encoder initialized successfully!" << std::endl;
      std::cout << "  Model: " << model_path << std::endl;
      std::cout << "  Input dimension: " << config_.input_dimension << std::endl;
      std::cout << "  Token dimension: " << config_.token_dimension << std::endl;
      std::cout << "  Input tensor: " << input_tensor_name_ << std::endl;
      std::cout << "  Output tensor: " << output_tensor_name_ << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "✗ EncoderEngine::Initialize - Exception: " << e.what() << std::endl;
      inference_engine_.reset();
      return false;
    }
  }


  /**
   * @brief Set input data for the encoder
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
   * @brief Run encoder inference and populate internal token buffer
   * @param stream CUDA stream for inference (uses internal stream if nullptr)
   * @return true if inference successful, false otherwise
   */
  bool Encode(cudaStream_t stream = nullptr) {
    if (!initialized_) { std::cerr << "✗ EncoderEngine::Encode - Not initialized" << std::endl; return false; }
    if (!inference_engine_) { std::cerr << "✗ EncoderEngine::Encode - TensorRT engine not initialized" << std::endl; return false; }

    cudaStream_t encode_stream = (stream != nullptr) ? stream : cuda_stream_;

    // Transfer input data from CPU to GPU
    inference_engine_->SetInputData(input_tensor_name_, encoder_input_buffer_);

    if (graph_captured_ && cuda_graph_exec_ != nullptr) {
      cudaError_t status = cudaGraphLaunch(cuda_graph_exec_, encode_stream);
      if (status != cudaSuccess) {
        std::cerr << "✗ EncoderEngine::Encode - Failed to launch CUDA graph: " << cudaGetErrorString(status) << std::endl;
        return false;
      }
    } else {
      if (!inference_engine_->Enqueue(encode_stream)) {
        std::cerr << "✗ EncoderEngine::Encode - Failed to enqueue inference" << std::endl;
        return false;
      }
    }
    
    // Automatically populate internal token buffer after inference (GPU to CPU)
    inference_engine_->GetOutputDataAsync(output_tensor_name_, token_buffer_, encode_stream);
    cudaStreamSynchronize(encode_stream);
    
    return true;
  }

  /**
   * @brief Capture CUDA graph for optimized execution
   * @return true if capture successful, false otherwise
   */
  bool CaptureGraph() {
    if (!initialized_ || !inference_engine_) {
      std::cerr << "✗ EncoderEngine::CaptureGraph - Cannot capture (engine not initialized)" << std::endl;
      return false;
    }
    if (graph_captured_) { 
      std::cout << "Encoder CUDA graph already captured" << std::endl;
      return true; 
    }

    std::cout << "Capturing encoder CUDA graph..." << std::endl;
    cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed);
    if (!inference_engine_->Enqueue(cuda_stream_)) {
      std::cerr << "✗ Failed to enqueue encoder inference for CUDA graph capture" << std::endl;
      cudaStreamEndCapture(cuda_stream_, &cuda_graph_);
      return false;
    }
    cudaStreamEndCapture(cuda_stream_, &cuda_graph_);
    cudaStreamSynchronize(cuda_stream_);
    
    cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, NULL, NULL, 0);
    
    graph_captured_ = true;
    std::cout << "✓ Encoder CUDA graph captured successfully!" << std::endl;
    return true;
  }

  /**
   * @brief Get CUDA stream used by encoder
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
   * @brief Get token dimension
   * @return Token dimension size
   */
  size_t GetTokenDimension() const { return config_.token_dimension; }

  /**
   * @brief Get current token source type
   * @return Token source type
   */
  // EncoderTokenSource GetTokenSource() const { return config_.source; } // Removed

  /**
   * @brief Set token source type
   * @param source Token source type
   */
  // void SetTokenSource(EncoderTokenSource source) { config_.source = source; } // Removed

  /**
   * @brief Check if encoder is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const { return initialized_; }

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
   * @brief Get reference to internal input buffer
   * @return Reference to input buffer
   */
  TPinnedVector<float>& GetInputBuffer() { return encoder_input_buffer_; }

  /**
   * @brief Get reference to internal token buffer
   * @return Reference to token buffer
   */
  TPinnedVector<float>& GetTokenBuffer() { return token_buffer_; }

  /**
   * @brief Destroy and clean up encoder resources
   */
  void Destroy() {
    if (!initialized_) { return; }
    if (cuda_graph_exec_ != nullptr) { cudaGraphExecDestroy(cuda_graph_exec_); cuda_graph_exec_ = nullptr; }
    if (cuda_graph_ != nullptr) { cudaGraphDestroy(cuda_graph_); cuda_graph_ = nullptr; }
    if (cuda_stream_ != nullptr) { cudaStreamDestroy(cuda_stream_); cuda_stream_ = nullptr; }
    if (inference_engine_) { inference_engine_->Destroy(); inference_engine_.reset(); }
    graph_captured_ = false; initialized_ = false;
  }

private:
  // Internal configuration state
  struct Config {
    std::string model_path;
    int device_id = 0;
    size_t input_dimension = 0;
    size_t token_dimension = 0;
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
  TPinnedVector<float> encoder_input_buffer_;

  // Token storage
  TPinnedVector<float> token_buffer_;

  // State
  bool initialized_ = false;
  bool graph_captured_ = false;
};

#endif // ENCODER_HPP
