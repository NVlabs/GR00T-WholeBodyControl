#pragma once

template<typename T>
void TRTInferenceEngine::SetInputData(const std::string& name, const T* data, size_t elementCount)
{
    SetInputData(name, static_cast<const void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void TRTInferenceEngine::SetInputData(const std::string& name, const TPinnedVector<T>& data)
{
    SetInputData(name, static_cast<const void*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void TRTInferenceEngine::GetOutputData(const std::string& name, T* data, size_t elementCount)
{
    GetOutputData(name, static_cast<void*>(data), sizeof(T) * elementCount);
}

template<typename T>
void TRTInferenceEngine::GetOutputData(const std::string& name, TPinnedVector<T>& data)
{
    GetOutputData(name, static_cast<void*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void TRTInferenceEngine::SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t stream)
{
    SetInputDataAsync(name, static_cast<const void*>(data), sizeof(T) * elementCount, stream);
}

template<typename T>
void TRTInferenceEngine::SetInputDataAsync(const std::string& name, const TPinnedVector<T>& data, cudaStream_t stream)
{
    SetInputDataAsync(name, static_cast<const void*>(data.data()), sizeof(T) * data.size(), stream);
}

template<typename T>
void TRTInferenceEngine::GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t stream)
{
    GetOutputDataAsync(name, static_cast<void*>(data), sizeof(T) * elementCount, stream);
}

template<typename T>
void TRTInferenceEngine::GetOutputDataAsync(const std::string& name, TPinnedVector<T>& data, cudaStream_t stream)
{
    GetOutputDataAsync(name, static_cast<void*>(data.data()), sizeof(T) * data.size(), stream);
}

