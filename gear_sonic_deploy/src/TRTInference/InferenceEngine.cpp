#include "InferenceEngine.h"
#include "Log.h"
#include "picosha2.h"

#include <fstream>
#include <array>
#include <bitset>
#include <vector>
#include <map>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace
{
    class Logger : public nvinfer1::ILogger
    {
    public:
        Logger( bool verbose )
            : m_flags( UINT8_MAX )
        {
            m_flags.set( static_cast<size_t>( Severity::kVERBOSE ), verbose );
        }

        void log( Severity severity, const char* msg ) noexcept override
        {
            if ( !m_flags.test( static_cast<size_t>( severity ) ) )
                return;

            switch ( severity )
            {
                case Severity::kINFO:
                case Severity::kVERBOSE:
                    LOG_INFO( msg );
                break;
                case Severity::kWARNING:
                    LOG_WARNING( msg );
                break;
                case Severity::kERROR:
                case Severity::kINTERNAL_ERROR:
                    LOG_ERROR( msg );
                break;
            }
        }

        void log( cudaError_t error )
        {
            if ( error == cudaError_t::cudaSuccess )
                return;

            LOG_ERROR( "CUDA failed with error [ " + std::to_string( error ) + " : " + cudaGetErrorName( error ) + " ]: " + cudaGetErrorString( error ) );
        }

    private:
        static constexpr uint8_t NB_FLAGS = static_cast<uint8_t>( Severity::kVERBOSE ) + 1;
        std::bitset<NB_FLAGS> m_flags;
    };

    Logger s_logger(false);

    bool FileExists( const std::string& filepath )
    {
        return std::ifstream( filepath.c_str() ).good();
    }

    std::string GetCudaDeviceName( int deviceID )
    {
        int deviceCount = 0;
        cudaGetDeviceCount( &deviceCount );

        if ( deviceID >= deviceCount )
            return std::string();

        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, deviceID );
        return std::string( prop.name );
    }

    int SizeOfTensorDataType( nvinfer1::DataType dataType )
    {
        int size = -1;
        switch ( dataType )
        {
            case nvinfer1::DataType::kINT64:
            size = sizeof( uint64_t );
            break;
            
            case nvinfer1::DataType::kINT32:
            case nvinfer1::DataType::kFLOAT:
            size = sizeof( uint32_t );
            break;
            
            case nvinfer1::DataType::kHALF:
            case nvinfer1::DataType::kBF16:
            size = sizeof( uint16_t );
            break;
            
            case nvinfer1::DataType::kFP8:
            case nvinfer1::DataType::kINT8:
            case nvinfer1::DataType::kBOOL:
            case nvinfer1::DataType::kUINT8:
            size = sizeof( uint8_t );
            break;
            
            default:
            break;
        }

        return size;
    }
}

bool ConvertONNXToTRT(
    const Options& options,
    const std::string& onnxModelPath,
    std::string& trtFile,
    const std::string prefix,
    bool forceConvert
)
{
    if ( !FileExists( onnxModelPath ) )
    {
        LOG_ERROR( "Cannot find ONNX mode at path: " + onnxModelPath );
        return false;
    }

    // Find a hash string identifying the onnx we're going to convert to 
    // tensorrt, plus the gpu we're going to convert it for:
    std::ifstream file(onnxModelPath, std::ios::binary);
    std::string trtHash = picosha2::hash256_hex_string(
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()
    );

    // hash in the cuda device name and precision::
    trtHash = picosha2::hash256_hex_string(trtHash + GetCudaDeviceName( options.deviceID ));
    trtHash = picosha2::hash256_hex_string(trtHash + std::to_string(int(options.precision)));

    // name of the tensorrt we're going to write to:
    const auto filenamePos = onnxModelPath.find_last_of( '/' ) + 1;
    trtFile = prefix + onnxModelPath.substr( filenamePos, onnxModelPath.find_last_of( '.' ) - filenamePos );
    trtFile += ".trt";

    if (!forceConvert && FileExists( trtFile ) )
    {
        std::ifstream file(trtFile, std::ios::binary);
        
        std::string hashInFile;
        hashInFile.resize(trtHash.size());

        file.read(const_cast<char*>(hashInFile.c_str()), trtHash.size());

        std::cerr << "hash value in file " <<  hashInFile << std::endl;
        std::cerr << "hash of onnx/gpu type " << trtHash << std::endl;

        if(hashInFile == trtHash)
        {
            return true;
        }
        LOG_INFO( "TRT file " + trtFile + " is out of date and needs to be regenerated. This could take a while..." );
    }
    else
    {
        LOG_INFO( "TRT file " + trtFile + " not found. This could take a while to generate..." );
    }


    // Create engine builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>( nvinfer1::createInferBuilder( s_logger ) );
    if ( !builder )
    {
        LOG_ERROR( "Could not create builder." );
        return false;
    }

    // Create network
    const uint32_t explicitBatch = 1U << static_cast<uint32_t>( nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH );
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>( builder->createNetworkV2( explicitBatch ) );
    if ( !network )
    {
        LOG_ERROR( "Could not create network." );
        return false;
    }

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>( nvonnxparser::createParser( *network, s_logger ) );
    if ( !parser )
    {
        LOG_ERROR( "Could not create ONNX parser." );
        return false;
    }

    // Read and parse the ONNX file
    auto parsed = parser->parseFromFile( onnxModelPath.c_str(), 0 );
    if ( !parsed )
    {
        LOG_ERROR( "Unable to parse the ONNX model at: " + onnxModelPath );
        return false;
    }

    // Ensure that all the inputs have the same batch size
    const int32_t inputCount = network->getNbInputs();
    if ( inputCount < 1 )
    {
        LOG_ERROR( "Model has no inputs." );
        return false;
    }

    // Set dynamic axis info for inputs:
    auto profile = builder->createOptimizationProfile();
    for ( int32_t i = 0; i < inputCount; ++i )
    {
        auto t = network->getInput( i );
        const auto name = t->getName();
        auto dims = t->getDimensions();

        bool hasDynamicAxes = false;
        auto dimsMin = dims;
        auto dimsOpt = dims;
        auto dimsMax = dims;

        for ( int32_t n = 0; n < dims.nbDims; ++n )
        {
            if ( dims.d[n] == -1 )
            {
                auto axis_sizes = options.defaultSizes;
                if (options.dynamic_axes_names.contains(t->getName()))
                {
                    auto& dynamic_axis_names = options.dynamic_axes_names.at(t->getName());
                    if (!dynamic_axis_names.contains(n))
                    {
                        LOG_ERROR("Tensor " + std::string(t->getName()) + " has a dynamic axes, but its name isn't specified in options.input_dynamic_axes");
                        return false;
                    }

                    auto axis_name = dynamic_axis_names.at(n);
                    if (!options.dynamic_axes_sizes.contains(axis_name))
                    {
                        LOG_ERROR("Named axis " + axis_name + " has no entry in options.dynamic_axes_sizes");
                        return false;
                    }
                    axis_sizes = options.dynamic_axes_sizes.at(axis_name);
                }

                hasDynamicAxes = true;
                dimsMin.d[n] = std::get<0>( axis_sizes );
                dimsOpt.d[n] = std::get<1>( axis_sizes );
                dimsMax.d[n] = std::get<2>( axis_sizes );
            }
        }
        profile->setDimensions( name, nvinfer1::OptProfileSelector::kMIN, dimsMin );
        profile->setDimensions( name, nvinfer1::OptProfileSelector::kOPT, dimsOpt );
        profile->setDimensions( name, nvinfer1::OptProfileSelector::kMAX, dimsMax );

        if ( t->isShapeTensor() )
        {
            if ( !options.shape_tensor_sizes.contains( name ) )
            {
                LOG_ERROR( "Tensor " + std::string( name ) + " controls the shape of an internal tensor and needs an entry in options.shape_tensor_sizes" );
                return false;
            }
            auto &sizes = options.shape_tensor_sizes.at( name );

            profile->setShapeValues( name, nvinfer1::OptProfileSelector::kMIN, std::get<0>( sizes ).data(), std::get<0>( sizes ).size());
            profile->setShapeValues( name, nvinfer1::OptProfileSelector::kOPT, std::get<1>( sizes ).data(), std::get<1>( sizes ).size() );
            profile->setShapeValues( name, nvinfer1::OptProfileSelector::kMAX, std::get<2>( sizes ).data(), std::get<2>( sizes ).size() );
        }
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>( builder->createBuilderConfig() );
    if ( !config )
    {
        LOG_ERROR( "Could not create builder config." );
        return false;
    }
    config->addOptimizationProfile( profile );

    // Set the precision level
    if ( options.precision == Precision::FP16 )
    {
        // Ensure the GPU supports FP16 inference
        if ( !builder->platformHasFastFp16() )
        {
            LOG_ERROR( "GPU does not support FP16 precision." );
            return false;
        }
        config->setFlag( nvinfer1::BuilderFlag::kFP16 );
    }

    // CUDA stream used for profiling by the builder.
    cudaStream_t stream;
    s_logger.log( cudaStreamCreate( &stream ) );
    config->setProfileStream( stream );

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the engine.
    // Doing so will provide you with more information on why exactly it is failing.
    auto serializedNetwork = builder->buildSerializedNetwork( *network, *config );
    if ( !serializedNetwork )
    {
        s_logger.log( cudaStreamDestroy( stream ) );

        LOG_ERROR( "Failed to build the engine!" );
        return false;
    }

    // Write the engine to disk
    std::ofstream outfile( trtFile, std::ofstream::binary );

    // Write the onnx/gpu hash first:
    outfile.write( trtHash.c_str(), trtHash.size() );

    // Write the engine:
    outfile.write( reinterpret_cast<const char*>( serializedNetwork->data() ), serializedNetwork->size() );

    LOG_INFO( "Saved TRT file to " + trtFile );
    s_logger.log( cudaStreamDestroy( stream ) );

    return true;
}

class TRTInferenceEngine::Impl
{
public:
    
    ~Impl()
    {
        Destroy();
    }

    bool Initialize( const std::string& trtPath, int deviceIndex, const Options::AxisNames& axisNames )
    {
        Destroy();

        // Read the serialized model from disk
        std::ifstream file( trtPath, std::ios::binary | std::ios::ate );
        std::streamsize size = file.tellg();
        if (size < 0)
        {
            LOG_ERROR("Unable to read engine file!");
            return false;
        }

        file.seekg( 0, std::ios::beg );
        std::vector<char> buffer( size );
        if ( !file.read( buffer.data(), size ) )
        {
            LOG_ERROR( "Unable to read engine file!" );
            return false;
        }

        // Create runtime to deserialize the engine file.
        m_runtime = std::unique_ptr<nvinfer1::IRuntime>( nvinfer1::createInferRuntime( s_logger ) );
        if ( !m_runtime )
        {
            LOG_ERROR( "Unable to create a runtime inference model." );
            return false;
        }

        // Set the device index
        if ( cudaSetDevice( deviceIndex ) != cudaError_t::cudaSuccess )
        {
            int gpuCount;
            cudaGetDeviceCount( &gpuCount );
            LOG_ERROR( "Unable to set GPU device index to " + std::to_string( deviceIndex ) + ". Current device has " + std::to_string( gpuCount ) + " CUDA - capable GPU(s)." );
            return false;
        }

        // Create an engine, a representation of the optimized model.
        // skip the first 64 bytes when deserializing the engine file as they just contain a hash:
        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>( m_runtime->deserializeCudaEngine( buffer.data() + 64, buffer.size() - 64 ) );
        if ( !m_engine )
        {
            LOG_ERROR("Unable to create engine.");
            return false;
        }

        // The execution context contains all of the state associated with a particular invocation
        m_context = std::unique_ptr<nvinfer1::IExecutionContext>( m_engine->createExecutionContext() );
        if ( !m_context )
        {
            LOG_ERROR("Unable to create execution context.");
            return false;
        }

        m_axis_names = axisNames;

        const uint32_t tensorCount = m_engine->getNbIOTensors();
        for (int32_t i = 0; i < tensorCount; ++i)
        {
            const auto tensorName = m_engine->getIOTensorName(i);
            const auto tensorIOMode = m_engine->getTensorIOMode(tensorName);
            m_dataTypeSizes[tensorName] = SizeOfTensorDataType(m_engine->getTensorDataType(tensorName));
        }
        
        return true;
    }


    bool InitInputs(const AxisSizes& axisSizes)
    {
        DestroyBuffers();

        const uint32_t tensorCount = m_engine->getNbIOTensors();
        for (int32_t i = 0; i < tensorCount; ++i)
        {
            const auto tensorName = m_engine->getIOTensorName(i);
            const auto tensorIOMode = m_engine->getTensorIOMode(tensorName);
            auto tensorShape = m_engine->getTensorShape(tensorName);
            const auto tensorDataType = m_engine->getTensorDataType(tensorName);

            for (int32_t n = 0; n < tensorShape.nbDims; ++n)
            {
                if (tensorShape.d[n] == -1)
                {
                    int64_t batchSize = 1;
                    if (m_axis_names.contains(tensorName))
                    {

                        auto& dynamic_axis_names = m_axis_names.at(tensorName);
                        if (!dynamic_axis_names.contains(n))
                        {
                            LOG_ERROR("Tensor " + std::string(tensorName) + " has a dynamic axes, but its name isn't specified in axisNames");
                            return false;
                        }

                        auto axis_name = dynamic_axis_names.at(n);
                        if (!axisSizes.contains(axis_name))
                        {
                            LOG_ERROR("Named axis " + axis_name + " has no entry in axisSizes");
                            return false;
                        }

                        batchSize = axisSizes.at(axis_name);
                    }
                    else if (axisSizes.size() > 0)
                    {
                        batchSize = axisSizes.begin()->second;
                    }

                    tensorShape.d[n] = batchSize;
                    if (tensorIOMode == nvinfer1::TensorIOMode::kINPUT)
                    {
                        auto batchMin = m_engine->getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMIN).d[n];
                        auto batchMax = m_engine->getProfileShape(tensorName, 0, nvinfer1::OptProfileSelector::kMAX).d[n];
                        if (batchSize < batchMin || batchSize > batchMax)
                        {
                            LOG_ERROR("Batch size " + std::to_string(batchSize) + " is outside of possible range [ " + std::to_string(batchMin) + ", " + std::to_string(batchMax) + "].");
                            return false;
                        }
                    }
                }
            }

            m_tensorShapes[tensorName] = tensorShape;
            static const std::string dataTypeStr[] = { "FLOAT", "HALF", "INT8", "INT32", "BOOL", "UINT8", "FP8", "BF16", "INT64" };
            static const std::string ioTypeStr[] = { "NONE", "INPUT", "OUTPUT" };
            std::string tensorInfo = ioTypeStr[int(tensorIOMode)] + " Tensor: " + std::string(tensorName) + " [";
            size_t elementCount = 1;
            for (int32_t d = 0; d < tensorShape.nbDims; ++d)
            {
                if (d != 0)
                    tensorInfo += ", ";

                tensorInfo += std::to_string(tensorShape.d[d]);

                if (elementCount > INT64_MAX / tensorShape.d[d])
                {
                    LOG_ERROR("Tensor too large: element count overflow");
                    return false;
                }
                elementCount *= tensorShape.d[d];
            }
            tensorInfo += "]. " + std::to_string(elementCount) + " elements of type " + dataTypeStr[int(tensorDataType)];
            LOG_INFO(tensorInfo);

            const auto sizeOfTensorData = SizeOfTensorDataType(tensorDataType);
            if (sizeOfTensorData == -1)
            {
                LOG_ERROR("Unknown tensor data type " + std::to_string((int)tensorDataType) + ".");
                return false;
            }

            void* cudaBuffer;
            size_t size = elementCount * sizeOfTensorData;
            auto err = cudaMalloc(&cudaBuffer, size);
            s_logger.log(err);
            if (err != cudaSuccess)
            {
                LOG_ERROR("Failed to allocate CUDA memory");
                return false;
            }

            s_logger.log(cudaMemset(cudaBuffer, 0, size));

            m_context->setTensorAddress(tensorName, cudaBuffer);
            if (tensorIOMode == nvinfer1::TensorIOMode::kINPUT)
            {
                m_inputBuffers[tensorName] = cudaBuffer;
            }
            else
            {
                m_outputBuffers[tensorName] = cudaBuffer;
            }
        }

        // Set input tensor shapes to specified batch size
        for (int i = 0; i < tensorCount; ++i)
        {
            const auto tensorName = m_engine->getIOTensorName(i);
            if (m_engine->getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kINPUT)
                continue;

            auto tensorShape = m_tensorShapes[tensorName];

            m_context->setInputShape(tensorName, tensorShape);
        }

        return true;
    }

    void DestroyBuffers()
    {
        for (auto& kv : m_inputBuffers)
        {
            s_logger.log(cudaFree(kv.second));
        }
        for (auto& kv : m_outputBuffers)
        {
            s_logger.log(cudaFree(kv.second));
        }

        m_inputBuffers.clear();
        m_outputBuffers.clear();
        m_tensorShapes.clear();
    }

    void Destroy()
    {
        DestroyBuffers();

        m_context.reset();
        m_engine.reset();
        m_runtime.reset();
    }

    bool ValidateTensorSize(const std::string& name, size_t byteCount) const
    {
        if (m_tensorShapes.empty())
        {
            LOG_ERROR("ValidateTensorSize: InitInputs has not been called.");
            return false;
        }

        if (!m_tensorShapes.contains(name))
        {
            LOG_ERROR("ValidateTensorSize: Tensor " + name + " not found.");
            return false;
        }

        size_t elementCount = 1;
        const auto& tensorShape = m_tensorShapes.at(name);
        for (int32_t d = 0; d < tensorShape.nbDims; ++d)
            elementCount *= tensorShape.d[d];

        const size_t expectedByteCount = elementCount * m_dataTypeSizes.at(name);
        if (expectedByteCount != byteCount)
        {
            LOG_ERROR(
                "Invalid data size for tensor [" + name + "] containing " + std::to_string(elementCount) + " elements. Expected " +
                std::to_string(expectedByteCount) + " bytes, got " + std::to_string(byteCount) + " bytes instead.");
            return false;
        }

        return true;
    }

    std::vector<std::string> GetInputBufferNames() const
    {
        std::vector<std::string> names;
        names.reserve(m_inputBuffers.size());

        for (const auto& pair : m_inputBuffers) 
        {
            names.push_back(pair.first);
        }
        return names;
    }

    std::vector<std::string> GetOutputBufferNames() const
    {
        std::vector<std::string> names;
        names.reserve(m_outputBuffers.size());

        for (const auto& pair : m_outputBuffers)
        {
            names.push_back(pair.first);
        }
        return names;
    }

    void SetInputData(const std::string& name, const void* data, size_t byteCount)
    {
        if(!ValidateTensorSize(name, byteCount))
        {
            LOG_ERROR("SetInputData: Tensor " + name + " is not valid.");
            return;
        }
        s_logger.log(cudaMemcpy(m_inputBuffers[name], data, byteCount, cudaMemcpyHostToDevice));
    }

    void SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t stream)
    {
        if(!ValidateTensorSize(name, byteCount))
        {
            LOG_ERROR("SetInputDataAsync: Tensor " + name + " is not valid.");
            return;
        }
        s_logger.log(cudaMemcpyAsync(m_inputBuffers[name], data, byteCount, cudaMemcpyHostToDevice, stream));
    }

    void GetOutputData(const std::string& name, void* data, size_t byteCount)
    {
        if(!ValidateTensorSize(name, byteCount))
        {
            LOG_ERROR("GetOutputData: Tensor " + name + " is not valid.");
            return;
        }
        s_logger.log(cudaMemcpy(data, m_outputBuffers[name], byteCount, cudaMemcpyDeviceToHost));
    }

    void GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t stream)
    {
        if(!ValidateTensorSize(name, byteCount))
        {
            LOG_ERROR("GetOutputDataAsync: Tensor " + name + " is not valid.");
            return;
        }
        s_logger.log(cudaMemcpyAsync(data, m_outputBuffers[name], byteCount, cudaMemcpyDeviceToHost, stream));
    }

    bool Enqueue( cudaStream_t stream )
    {
        if (m_tensorShapes.empty())
        {
            LOG_ERROR("Enqueue: InitInputs has not been called.");
            return false;
        }

        if (!m_context->allInputDimensionsSpecified())
        {
            LOG_ERROR("Inference engine inputs aren't defined.");
            return false;
        }

        if(!m_context->enqueueV3(stream))
        {
            LOG_ERROR("Enqueue: Failed to enqueue.");
            return false;
        }
        return true;
    }

    bool GetTensorShape(std::string name, std::vector<int64_t>& shape) const
    {
        if (m_tensorShapes.empty())
        {
            LOG_ERROR("ValidateTensorSize: InitInputs has not been called.");
            return false;
        }

        if (!m_tensorShapes.contains(name))
        {
            LOG_ERROR("ValidateTensorSize: Tensor " + name + " not found.");
            return false;
        }

        const auto& tensorShape = m_tensorShapes.at(name);
        shape.clear();
        for (int dim=0; dim < tensorShape.nbDims; ++dim)
        {
            shape.push_back(tensorShape.d[dim]);
        }
        return true;
    }

    nvinfer1::DataType GetTensorDataType(std::string name) const
    {
        auto tensorDataType = m_engine->getTensorDataType(name.c_str());
        return tensorDataType;
    }

    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;

    std::map<std::string, nvinfer1::Dims> m_tensorShapes;

    std::map<std::string, void*> m_inputBuffers;
    std::map<std::string, void*> m_outputBuffers;

    std::map<std::string, int> m_dataTypeSizes;

    Options::AxisNames m_axis_names;
};

TRTInferenceEngine::TRTInferenceEngine() : m_impl( new Impl )
{
}

TRTInferenceEngine::~TRTInferenceEngine()
{
}

bool TRTInferenceEngine::Initialize(const std::string& trtPath, int deviceIndex, const Options::AxisNames& axisNames)
{
    return m_impl->Initialize(trtPath, deviceIndex, axisNames);
}

bool TRTInferenceEngine::InitInputs(const AxisSizes& axisSizes)
{
    return m_impl->InitInputs(axisSizes);
}

void TRTInferenceEngine::Destroy()
{
    m_impl->Destroy();
}

std::vector<std::string> TRTInferenceEngine::GetInputTensorNames() const
{
    return m_impl->GetInputBufferNames();
}

std::vector<std::string> TRTInferenceEngine::GetOutputTensorNames() const
{
    return m_impl->GetOutputBufferNames();
}

void TRTInferenceEngine::SetInputData(const std::string& name, const void* data, size_t byteCount)
{
    m_impl->SetInputData(name, data, byteCount);
}

void TRTInferenceEngine::GetOutputData(const std::string& name, void* data, size_t byteCount)
{
    m_impl->GetOutputData( name, data, byteCount);
}

void TRTInferenceEngine::SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t stream)
{
    m_impl->SetInputDataAsync(name, data, byteCount, stream);
}

void TRTInferenceEngine::GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t stream)
{
    m_impl->GetOutputDataAsync(name, data, byteCount, stream);
}

bool TRTInferenceEngine::Enqueue(cudaStream_t stream)
{
    return m_impl->Enqueue(stream);
}

bool TRTInferenceEngine::GetTensorShape(std::string name, std::vector<int64_t>& shape) const
{
    return m_impl->GetTensorShape(name, shape);
}

DataType TRTInferenceEngine::GetTensorDataType(std::string name) const
{
    auto dataType = m_impl->GetTensorDataType(name);
    switch (dataType)
    {
        case nvinfer1::DataType::kFLOAT:
        return DataType::FLOAT;
        case nvinfer1::DataType::kHALF:
        return DataType::HALF;
        case nvinfer1::DataType::kINT8:
        return DataType::INT8;
        case nvinfer1::DataType::kINT32:
        return DataType::INT32;
        case nvinfer1::DataType::kBOOL:
        return DataType::BOOL;
        case nvinfer1::DataType::kUINT8:
        return DataType::UINT8;
        case nvinfer1::DataType::kINT64:
        return DataType::INT64;
        default:
        LOG_ERROR("ValidateTensorSize: Tensor " + name + " not found.");
        return DataType::UNKNOWN;
    }
}