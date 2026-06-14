#pragma once

#include "Utility.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

enum class Precision
{
    FP32,
    FP16
};

enum class DataType
{
    FLOAT,
    HALF,
    INT8,
    INT32,
    BOOL,
    UINT8,
    INT64,
    UNKNOWN
};


struct Options
{
    using AxisNames = std::map< std::string, std::map<int, std::string> >;
    using AxisSizes = std::map<std::string, std::tuple<int, int, int> >;
    using ShapeTensorSizes = std::map<std::string, std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> >;

    Precision        precision = Precision::FP32;
    AxisNames        dynamic_axes_names;
    AxisSizes        dynamic_axes_sizes;
    ShapeTensorSizes shape_tensor_sizes;
    std::tuple<int, int, int> defaultSizes = { 1,8,16 };
    int              deviceID = 0;
};

bool ConvertONNXToTRT(
    const Options& options,
    const std::string& onnxModelPath,
    std::string& generatedTRTFile,
    const std::string prefix = "",
    bool forceConvert = false
);

// Forward declaration
typedef struct CUstream_st *cudaStream_t;

class TRTInferenceEngine
{
public:
    TRTInferenceEngine();
    ~TRTInferenceEngine();

    using AxisSizes = std::map<std::string, int >;
    bool Initialize(const std::string& trtPath, int deviceID, const Options::AxisNames& axisNames = {});
    bool InitInputs(const AxisSizes& axisSizes = {});
    void Destroy();

    void SetInputData(const std::string& name, const void* data, size_t byteCount);
    template<typename T> void SetInputData(const std::string& name, const T* data, size_t elementCount);
    template<typename T> void SetInputData(const std::string& name, const TPinnedVector<T>& data);

    void GetOutputData(const std::string& name, void* data, size_t byteCount);
    template<typename T> void GetOutputData(const std::string& name, T* data, size_t elementCount);
    template<typename T> void GetOutputData(const std::string& name, TPinnedVector<T>& data);

    void SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t stream);
    template<typename T> void SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t stream);
    template<typename T> void SetInputDataAsync(const std::string& name, const TPinnedVector<T>& data, cudaStream_t stream);

    void GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t stream);
    template<typename T> void GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t stream);
    template<typename T> void GetOutputDataAsync(const std::string& name, TPinnedVector<T>& data, cudaStream_t stream);

    std::vector<std::string> GetInputTensorNames() const;
    std::vector<std::string> GetOutputTensorNames() const;

    bool GetTensorShape(std::string name, std::vector<int64_t>& shape) const;
    DataType GetTensorDataType(std::string name) const;
    bool Enqueue(cudaStream_t stream);

private:
    class Impl;
    std::shared_ptr<Impl> m_impl = nullptr;
};

#include "InferenceEngine.inl"
