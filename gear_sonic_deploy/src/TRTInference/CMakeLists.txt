message(STATUS "blah = ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")

set(SRC_FILES
    "InferenceEngine.cpp"
    "InferenceEngine.h"
    "InferenceEngine.inl"
    "Utility.h"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(TRTInference ${SRC_FILES})

# Explicitly set C++ standard for this library
set_target_properties(TRTInference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Set up include directories
set(CUDA_INCLUDES "")
if(CUDAToolkit_FOUND)
    set(CUDA_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
elseif(CUDA_INCLUDE_DIR)
    set(CUDA_INCLUDES ${CUDA_INCLUDE_DIR})
endif()

target_include_directories(TRTInference
    # InferenceEngine.h is part of this target's public API. Consumers include
    # it as <TRTInference/InferenceEngine.h>, so the project src root must
    # propagate with the linked target. Utility.h also exposes CUDA headers.
    PUBLIC ${PROJECT_SOURCE_DIR}/src ${CUDA_INCLUDES}
    PRIVATE ${TensorRT_INCLUDE_DIRS}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(TRTInference PRIVATE -Wno-deprecated-declarations -std=c++20)
endif()

target_link_libraries(TRTInference ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
