cmake_minimum_required(VERSION 3.14 FATAL_ERROR) # Set the minimum required
                                                 # version of CMake

project(
  g1_deploy
  VERSION 1.0.0
  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

# Library target
add_library(${PROJECT_NAME} INTERFACE)

include(CheckLanguage)

# Optional builds
option(BUILD_SRCS "Build sources" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(TensorRT REQUIRED)
find_package(Threads REQUIRED)
find_package(onnxruntime REQUIRED)
# Find CUDA Toolkit with proper environment variable support
if(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDAToolkit_ROOT $ENV{CUDAToolkit_ROOT})
    message(STATUS "🎯 Using CUDAToolkit_ROOT from environment: ${CUDAToolkit_ROOT}")
endif()

if(DEFINED ENV{CUDA_HOME})
    message(STATUS "🔍 Using CUDA_HOME from environment: $ENV{CUDA_HOME}")
    list(APPEND CMAKE_PREFIX_PATH $ENV{CUDA_HOME})
endif()

find_package(CUDAToolkit 10.2 QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "✅ CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
    
    # Report CUDA architecture and paths for debugging
    message(STATUS "   CUDA Root: ${CUDAToolkit_TARGET_DIR}")
    message(STATUS "   CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")
    
    # For CUDA 12.6+, enable newer features if available
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.6")
        message(STATUS "   🚀 CUDA 12.6+ detected - enabling optimized features")
        # Add any CUDA 12.6+ specific optimizations here
        add_compile_definitions(CUDA_12_6_PLUS=1)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
        message(STATUS "   🔧 CUDA 12.x detected")
        add_compile_definitions(CUDA_12_PLUS=1)
    endif()
else()
    # Fallback: Look for CUDA runtime libraries and headers
    message(STATUS "🔍 CUDA Toolkit not found via find_package, searching for runtime libraries...")
    message(STATUS "   This is normal for systems with runtime-only CUDA installations")
    
    # Debug: Show environment variables
    if(DEFINED ENV{CUDAToolkit_ROOT})
        message(STATUS "   🔍 Environment CUDAToolkit_ROOT: $ENV{CUDAToolkit_ROOT}")
    endif()
    if(DEFINED ENV{CUDA_HOME})
        message(STATUS "   🔍 Environment CUDA_HOME: $ENV{CUDA_HOME}")
    endif()
    
    # Enhanced search with better path prioritization - now includes dynamic version detection
    # First, build dynamic path lists for any CUDA versions
    file(GLOB CUDA_SBSA_PATHS "/usr/local/cuda*/targets/sbsa-linux/lib")
    file(GLOB CUDA_AARCH64_PATHS "/usr/local/cuda*/targets/aarch64-linux/lib") 
    file(GLOB CUDA_LIB64_PATHS "/usr/local/cuda*/lib64")
    
    find_library(CUDA_RUNTIME_LIBRARY
        NAMES cudart libcudart
        HINTS
            # Environment variable paths
            $ENV{CUDAToolkit_ROOT}/targets/sbsa-linux/lib
            $ENV{CUDAToolkit_ROOT}/targets/aarch64-linux/lib
            $ENV{CUDAToolkit_ROOT}/lib64
            $ENV{CUDAToolkit_ROOT}/lib
            $ENV{CUDA_HOME}/targets/sbsa-linux/lib
            $ENV{CUDA_HOME}/targets/aarch64-linux/lib
            $ENV{CUDA_HOME}/lib64
            $ENV{CUDA_HOME}/lib
        PATHS
            # Dynamic CUDA version paths (automatically found)
            ${CUDA_SBSA_PATHS}
            ${CUDA_AARCH64_PATHS}
            ${CUDA_LIB64_PATHS}
            # Static fallback paths
            /usr/local/cuda/targets/sbsa-linux/lib
            /usr/local/cuda/targets/aarch64-linux/lib
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
            # System library paths
            /usr/lib/aarch64-linux-gnu
            /usr/lib/x86_64-linux-gnu
            /usr/lib64
            /usr/lib
        NO_DEFAULT_PATH
        )
    
    # Build dynamic include paths for any CUDA versions
    file(GLOB CUDA_SBSA_INCLUDE_PATHS "/usr/local/cuda*/targets/sbsa-linux/include")
    file(GLOB CUDA_AARCH64_INCLUDE_PATHS "/usr/local/cuda*/targets/aarch64-linux/include")
    file(GLOB CUDA_INCLUDE_PATHS "/usr/local/cuda*/include")
    
    find_path(CUDA_INCLUDE_DIR cuda_runtime.h
        HINTS
            # Environment variable paths
            $ENV{CUDAToolkit_ROOT}/targets/sbsa-linux/include
            $ENV{CUDAToolkit_ROOT}/targets/aarch64-linux/include
            $ENV{CUDAToolkit_ROOT}/include
            $ENV{CUDA_HOME}/targets/sbsa-linux/include
            $ENV{CUDA_HOME}/targets/aarch64-linux/include
            $ENV{CUDA_HOME}/include
        PATHS
            # Dynamic CUDA version paths (automatically found)
            ${CUDA_SBSA_INCLUDE_PATHS}
            ${CUDA_AARCH64_INCLUDE_PATHS}
            ${CUDA_INCLUDE_PATHS}
            # Static fallback paths
            /usr/local/cuda/targets/sbsa-linux/include
            /usr/local/cuda/targets/aarch64-linux/include
            /usr/local/cuda/include
            # System include paths
            /usr/include
            /usr/include/cuda
        PATH_SUFFIXES
            cuda
        NO_DEFAULT_PATH
        )
    
    # Debug: Check if files exist in Jetson-specific locations
    message(STATUS "🔍 Manually checking Jetson CUDA locations...")
    
    # Check for libcudart with version number (as found in your system)
    if(EXISTS "/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.12.6.77")
        message(STATUS "✅ Found libcudart.so.12.6.77 in Jetson sbsa-linux location!")
        set(CUDA_RUNTIME_LIBRARY "/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.12.6.77")
    elseif(EXISTS "/usr/local/cuda/targets/sbsa-linux/lib")
        # Check if any libcudart exists in sbsa-linux directory
        file(GLOB CUDART_FILES "/usr/local/cuda/targets/sbsa-linux/lib/libcudart*")
        if(CUDART_FILES)
            list(GET CUDART_FILES 0 FIRST_CUDART)
            message(STATUS "✅ Found CUDA runtime in sbsa-linux: ${FIRST_CUDART}")
            set(CUDA_RUNTIME_LIBRARY ${FIRST_CUDART})
        endif()
    elseif(EXISTS "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
        message(STATUS "✅ Found libcudart.so in Jetson aarch64-linux location!")
        set(CUDA_RUNTIME_LIBRARY "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so")
    elseif(EXISTS "/usr/local/cuda/lib64/libcudart.so")
        message(STATUS "✅ Found libcudart.so in standard location!")
        set(CUDA_RUNTIME_LIBRARY "/usr/local/cuda/lib64/libcudart.so")
    else()
        message(STATUS "❌ libcudart.so not found in expected locations")
    endif()
    
    if(EXISTS "/usr/local/cuda/targets/sbsa-linux/include/cuda_runtime.h")
        message(STATUS "✅ Found cuda_runtime.h in Jetson sbsa-linux location!")
        set(CUDA_INCLUDE_DIR "/usr/local/cuda/targets/sbsa-linux/include")
    elseif(EXISTS "/usr/local/cuda/targets/aarch64-linux/include/cuda_runtime.h")
        message(STATUS "✅ Found cuda_runtime.h in Jetson aarch64-linux location!")
        set(CUDA_INCLUDE_DIR "/usr/local/cuda/targets/aarch64-linux/include")
    elseif(EXISTS "/usr/local/cuda/include/cuda_runtime.h")
        message(STATUS "✅ Found cuda_runtime.h in standard location!")
        set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
    else()
        message(STATUS "❌ cuda_runtime.h not found in expected locations")
    endif()
    
    if(CUDA_RUNTIME_LIBRARY AND CUDA_INCLUDE_DIR)
        message(STATUS "✅ CUDA runtime components verified")
        
        # Try to detect CUDA version from headers
        if(EXISTS "${CUDA_INCLUDE_DIR}/cuda.h")
            file(STRINGS "${CUDA_INCLUDE_DIR}/cuda.h" CUDA_VERSION_DEFINES
                REGEX "#define CUDA_VERSION ")
            if(CUDA_VERSION_DEFINES MATCHES "#define CUDA_VERSION ([0-9]+)")
                math(EXPR CUDA_VERSION_MAJOR "${CMAKE_MATCH_1} / 1000")
                math(EXPR CUDA_VERSION_MINOR "(${CMAKE_MATCH_1} % 1000) / 10")
                set(CUDA_VERSION_STRING "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
                message(STATUS "   Detected CUDA version: ${CUDA_VERSION_STRING}")
                
                # Enable version-specific features
                if(CUDA_VERSION_MAJOR GREATER_EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 6)
                    message(STATUS "   🚀 CUDA 12.6+ detected - enabling optimized features")
                    add_compile_definitions(CUDA_12_6_PLUS=1)
                elseif(CUDA_VERSION_MAJOR GREATER_EQUAL 12)
                    message(STATUS "   🔧 CUDA 12.x detected")
                    add_compile_definitions(CUDA_12_PLUS=1)
                endif()
            endif()
        endif()
        
        # Create imported target for compatibility with modern CMake
        add_library(CUDA::cudart SHARED IMPORTED)
        set_target_properties(CUDA::cudart PROPERTIES
            IMPORTED_LOCATION "${CUDA_RUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIR}")
    else()
        message(STATUS "❌ CUDA components not found:")
        if(NOT CUDA_RUNTIME_LIBRARY)
            message(STATUS "   - CUDA runtime library (libcudart) not found")
        endif()
        if(NOT CUDA_INCLUDE_DIR)
            message(STATUS "   - CUDA headers (cuda_runtime.h) not found")
        endif()
        message(FATAL_ERROR "
🚨 CUDA installation incomplete. Please run: 
   ./scripts/install_deps.sh

💡 For Jetson systems with CUDA 12.6:
   - Ensure JetPack SDK development components are installed
   - Check if CUDA is installed at /usr/local/cuda-12.6/
   - Run 'ls -la /usr/local/cuda*' to verify CUDA installation")
    endif()
endif()
find_package(ZLIB REQUIRED)

# Use local unitree_sdk2 subfolder
# Disable building examples for unitree_sdk2
set(BUILD_EXAMPLES
    OFF
    CACHE BOOL "Build unitree_sdk2 examples" FORCE)

add_subdirectory(thirdparty/unitree_sdk2)

# Add unitree_sdk2 include directories
target_include_directories(${PROJECT_NAME}
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/unitree_sdk2/include)

# Link to unitree_sdk2 which includes all necessary dependencies
target_link_libraries(${PROJECT_NAME} INTERFACE unitree_sdk2)

# Examples
if(BUILD_SRCS)
  add_subdirectory(src)
endif()
