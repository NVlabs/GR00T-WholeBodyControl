/**
 * @file ort_session.hpp
 * @deprecated This file is NOT used in the current build.  It is only included
 *             by localmotion_kplanner_onnx.hpp, which is itself unused.
 *             Both files are kept for reference only.
 *
 * @brief Lightweight wrapper around an ONNX Runtime inference session.
 *
 * OrtSession loads an ONNX model file, auto-discovers all input / output
 * tensor names and shapes at construction time, and exposes a simple `Run()`
 * method for synchronous CPU inference.
 */

#ifndef ORT_SESSION_HPP
#define ORT_SESSION_HPP

#include <string>
#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>

/**
 * @class OrtSession
 * @brief Thin RAII wrapper around `Ort::Session` with auto-discovered metadata.
 *
 * On construction, iterates over every model input and output to record
 * names, shapes, and element types.  Provides two `Run()` overloads:
 *  - One that returns `std::vector<Ort::Value>` (allocating new outputs).
 *  - One that writes into pre-allocated output tensors (in-place).
 */
class OrtSession {
public:
    /**
     * @brief Constructs an OrtSession and initializes the ONNX Runtime session
     * 
     * @param model_path Path to the ONNX model file
     * @param env Reference to the ONNX Runtime environment
     * @param allocator Reference to the ONNX Runtime allocator for memory management
     * 
     * This constructor:
     * 1. Creates an ONNX Runtime session with optimized settings
     * 2. Queries and stores input/output metadata from the model
     * 3. Prepares internal data structures for efficient inference
     */
    OrtSession(std::string model_path, Ort::Env &env, Ort::AllocatorWithDefaultOptions &allocator) :
        m_session(env, model_path.c_str(), Ort::SessionOptions {nullptr}),
        m_model_path(model_path)
    {
        // Configure session options for optimal performance
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);  // Use single thread for intra-op parallelism
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);  // Enable extended optimizations

        // Discover and catalog all input nodes from the ONNX model
        size_t num_input_nodes = m_session.GetInputCount();
        std::cout << "Model has " << num_input_nodes << " inputs:" << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++)
        {
            // Retrieve the input node name
            auto input_name = m_session.GetInputNameAllocated(i, allocator);
            std::string name_str(input_name.get());

            // Extract tensor type and shape information
            auto input_type_info = m_session.GetInputTypeInfo(i);
            auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();

            // Store input dimensions for runtime tensor creation
            m_input_node_dims.push_back(shape);

            std::cout << "  Input " << i << ": " << name_str << " - Shape: [";
            for (size_t j = 0; j < shape.size(); j++)
            {
                if (shape[j] == -1)
                {
                    std::cout << "dynamic";
                }
                else
                {
                    std::cout << shape[j];
                }
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "] - Type: " << tensor_info.GetElementType() << std::endl;

            // Store input node name for later reference
            m_input_node_names_str.push_back(name_str);
        }
  
        // Discover and catalog all output nodes from the ONNX model
        size_t num_output_nodes = m_session.GetOutputCount();
        std::cout << "Model has " << num_output_nodes << " outputs:" << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++)
        {
            // Retrieve the output node name
            auto output_name = m_session.GetOutputNameAllocated(i, allocator);
            std::string name_str(output_name.get());

            // Extract tensor type and shape information
            auto output_type_info = m_session.GetOutputTypeInfo(i);
            auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            m_output_node_dims.push_back(shape);

            std::cout << "  Output " << i << ": " << name_str << " - Shape: [";
            for (size_t j = 0; j < shape.size(); j++)
            {
                if (shape[j] == -1)
                {
                    std::cout << "dynamic";
                }
                else
                {
                    std::cout << shape[j];
                }
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "] - Type: " << tensor_info.GetElementType() << std::endl;

            // Store output node name for later reference
            m_output_node_names_str.push_back(name_str);
        }

        // Convert string vectors to const char* vectors required by ONNX Runtime API
        // This is necessary because ONNX Runtime expects null-terminated C-style strings
        for (const auto& name : m_input_node_names_str) { m_input_node_names.push_back(name.c_str()); }
        for (const auto& name : m_output_node_names_str) { m_output_node_names.push_back(name.c_str()); }
  
    }

    /**
     * @brief Executes inference on the loaded ONNX model
     * 
     * @param input_tensors Vector of input tensors containing the data to process
     * @return Vector of output tensors containing the inference results
     * 
     * This method performs a single inference pass through the model using the provided
     * input tensors. The input tensors must match the model's expected input specification
     * in terms of count, names, shapes, and data types.
     */
    std::vector<Ort::Value> Run(const std::vector<Ort::Value> &input_tensors)
    {
        return m_session.Run(Ort::RunOptions {nullptr}, m_input_node_names.data(), input_tensors.data(),
                                  input_tensors.size(), m_output_node_names.data(), m_output_node_names.size());
    }


    void Run(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors)
    {
        m_session.Run(
            Ort::RunOptions {nullptr},
            m_input_node_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            m_output_node_names.data(),
            output_tensors.data(),
            output_tensors.size()
        );
    }

    /**
     * @brief Gets the input node dimensions discovered from the model
     * 
     * @return Const reference to vector containing shape information for each input node
     * 
     * Each element in the returned vector corresponds to one input node and contains
     * the shape dimensions as int64_t values. Dynamic dimensions are represented as -1.
     */
    const std::vector<std::vector<int64_t>> &get_input_node_dims() const {
        return m_input_node_dims;
    }

    const std::vector<std::vector<int64_t>> &get_output_node_dims() const {
        return m_output_node_dims;
    }

    const std::vector<std::string> &get_input_node_names_str() const {
        return m_input_node_names_str;
    }

private:
    // Core ONNX Runtime session object that manages the loaded model
    Ort::Session m_session;
    
    // Storage for input and output node names as std::string objects
    std::vector<std::string> m_input_node_names_str;   ///< Input node names stored as strings
    std::vector<std::string> m_output_node_names_str;  ///< Output node names stored as strings
    
    // C-style string pointers required by ONNX Runtime API (point into above string vectors)
    std::vector<const char*> m_input_node_names;       ///< Input node names as C-style strings for runtime
    std::vector<const char*> m_output_node_names;      ///< Output node names as C-style strings for runtime
    
    // Shape information for input tensors, indexed by input node
    std::vector<std::vector<int64_t>> m_input_node_dims;  ///< Dimensions for each input node (-1 for dynamic dims)
    std::vector<std::vector<int64_t>> m_output_node_dims;  ///< Dimensions for each output node (-1 for dynamic dims)

    // Buffer for observation data (currently unused in this implementation)
    std::vector<double> m_obs_buffer;                   ///< Reserved buffer for observation processing

    // Path to the loaded model file for reference
    std::string m_model_path;                          ///< File path of the loaded ONNX model
};

#endif
