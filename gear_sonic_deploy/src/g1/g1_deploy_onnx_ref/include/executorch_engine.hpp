/**
 * @file executorch_engine.hpp
 * @brief ExecuTorch runner for `.pte` models (observations → outputs).
 *
 * A `.pte` file exported with Torch-TensorRT already holds the TensorRT engine,
 * so nothing is converted or cached at startup. PolicyEngine and EncoderEngine
 * use this class instead of TensorRT when their model path ends in `.pte`.
 *
 * ## I/O Contract
 *
 *   - **Input**: one float32 tensor, read from a host buffer.
 *   - **Output**: one float32 tensor, copied into a host buffer.
 *
 * `-DUSE_EXECUTORCH=ON` enables it in `run_tests` only: the robot program links
 * TensorRT 10, so it cannot run the TensorRT 11 engines in a `.pte` yet.
 * Elsewhere, loading a `.pte` fails with an error message.
 */

#ifndef EXECUTORCH_ENGINE_HPP
#define EXECUTORCH_ENGINE_HPP

#include <iostream>
#include <string>
#include <vector>

#if HAS_EXECUTORCH
#include <cstring>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#endif

/// True if the model path names an ExecuTorch program.
inline bool IsExecuTorchModel(const std::string& model_path) { return model_path.ends_with(".pte"); }

/**
 * @class ExecuTorchEngine
 * @brief Loads a `.pte` program and runs its `forward` method on host buffers.
 */
class ExecuTorchEngine {
public:
  /**
   * @brief Load the program and check it has one float32 input and one float32 output, both
   *        contiguous and in host memory
   * @param model_path Path to the `.pte` file
   * @return true if the model is loaded and has the expected I/O, false otherwise
   */
  bool Initialize(const std::string& model_path) {
    input_dimension_ = 0;
    output_dimension_ = 0;
#if HAS_EXECUTORCH
    module_ = std::make_unique<executorch::extension::Module>(model_path);
    const auto error = module_->load_method("forward");
    if (error != executorch::runtime::Error::Ok) {
      // Fails for a .pte built for another GPU type or TensorRT version, or when another
      // TensorRT version is already loaded in the process.
      std::cerr << "✗ Failed to load ExecuTorch program (" << executorch::runtime::to_string(error)
                << "): " << model_path << std::endl;
      module_.reset();
      return false;
    }
    auto meta = module_->method_meta("forward");
    if (!meta.ok()) {
      std::cerr << "✗ Cannot read the forward method of ExecuTorch program: " << model_path << std::endl;
      module_.reset();
      return false;
    }
    if (meta->num_inputs() != 1 || meta->num_outputs() != 1) {
      std::cerr << "✗ ExecuTorch program must have exactly 1 input and 1 output: " << model_path << std::endl;
      module_.reset();
      return false;
    }
    auto input = meta->input_tensor_meta(0);
    auto output = meta->output_tensor_meta(0);
    // The host buffers are flat, so other layouts (dim orders) would read and write the wrong elements.
    const auto is_flat_float_on_host = [](const executorch::runtime::TensorInfo& info) {
      return info.scalar_type() == executorch::aten::ScalarType::Float && info.device().is_cpu() &&
             executorch::runtime::is_contiguous_dim_order(info.dim_order().data(), info.dim_order().size());
    };
    if (!input.ok() || !output.ok() || !is_flat_float_on_host(*input) || !is_flat_float_on_host(*output)) {
      std::cerr << "✗ ExecuTorch program input and output must be contiguous float32 tensors in host memory: "
                << model_path << std::endl;
      module_.reset();
      return false;
    }
    input_sizes_.assign(input->sizes().begin(), input->sizes().end());
    input_dimension_ = input->nbytes() / sizeof(float);
    output_dimension_ = output->nbytes() / sizeof(float);
    return true;
#else
    std::cerr << "✗ Cannot load " << model_path
              << ": this build cannot run .pte models (see the ExecuTorch export section of the training guide)"
              << std::endl;
    return false;
#endif
  }

  /**
   * @brief Run the model
   * @param input Host buffer with GetInputDimension() floats
   * @param output Host buffer for GetOutputDimension() floats
   * @return true if inference successful, false otherwise
   */
  bool Run(const float* input, float* output) {
#if HAS_EXECUTORCH
    if (!module_) {
      std::cerr << "✗ ExecuTorchEngine::Run - Not initialized" << std::endl;
      return false;
    }
    // from_blob needs a mutable pointer; forward only reads the input.
    auto tensor = executorch::extension::from_blob(const_cast<float*>(input), input_sizes_);
    auto result = module_->forward(tensor);
    if (!result.ok()) {
      std::cerr << "✗ ExecuTorch forward failed" << std::endl;
      return false;
    }
    if (result->size() != 1 || !result->at(0).isTensor() ||
        static_cast<size_t>(result->at(0).toTensor().numel()) != output_dimension_) {
      std::cerr << "✗ ExecuTorch forward returned an unexpected output" << std::endl;
      return false;
    }
    std::memcpy(output, result->at(0).toTensor().const_data_ptr<float>(), output_dimension_ * sizeof(float));
    return true;
#else
    (void)input;
    (void)output;
    return false;
#endif
  }

  /**
   * @brief Run the model twice before the control loop
   *
   * The TensorRT delegate warms up on the first run. The second run records the CUDA graph that
   * use_cuda_graphs turned on at export, so later runs only replay it.
   * @param input Host buffer with GetInputDimension() floats
   * @return true if both runs succeeded, false otherwise
   */
  bool Warmup(const float* input) {
    std::vector<float> scratch(output_dimension_);
    return Run(input, scratch.data()) && Run(input, scratch.data());
  }

  /// Number of floats in the input tensor
  size_t GetInputDimension() const { return input_dimension_; }
  /// Number of floats in the output tensor
  size_t GetOutputDimension() const { return output_dimension_; }

private:
#if HAS_EXECUTORCH
  std::unique_ptr<executorch::extension::Module> module_;
  std::vector<executorch::aten::SizesType> input_sizes_;
#endif
  size_t input_dimension_ = 0;
  size_t output_dimension_ = 0;
};

#endif // EXECUTORCH_ENGINE_HPP
